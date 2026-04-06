# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics

from controling.control_module import myhier, relabel_choice_mask, bpr_loss, bpr_loss_factor, \
    suppose_first_classify_is_right, label_51_to_3_overlap,\
    use_bias, use_gt_box, use_labeled_box, bpr_mask
from controling.control_module import use_split, use_pattern_bias
from controling.Myhier_function import predicate_include_num
from controling.control_module import use_weight, weight

import random

class Multi_Label_Loss(nn.Module):

    def __init__(self, mode='normal', multi_matrix=None):
        super(Multi_Label_Loss, self).__init__()
        self.count_mode = mode
        self.one_hot_matrix = torch.from_numpy(multi_matrix).cuda()
        assert self.one_hot_matrix is not None

    def forward(self, input, target):
        '''method 2 faster'''
        logpt = F.log_softmax(input)
        count_matrix = self.one_hot_matrix[target].to(logpt.dtype)

        loss_ini = -torch.mul(logpt, count_matrix)
        loss_sum = loss_ini.sum(dim=1)

        return loss_sum.mean()

@registry.ROI_RELATION_PREDICTOR.register("HierMotifsE2E")
class HierMotifsE2E(nn.Module):
    def __init__(self, config, in_channels):
        super(HierMotifsE2E, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = use_bias

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            self.freq_bias = FrequencyBias(config, statistics)

        ''' xainjing '''
        self.hier_num = len(myhier)
        self.rel_classifer_all = self.generate_muti_networks()
        self.onehot_for_51 = self.generate_one_hot_vector(myhier)
        self.multi_label_loss = Multi_Label_Loss(mode='normal', multi_matrix=self.onehot_for_51)

        self.pattern_loss = nn.CrossEntropyLoss()
        self.second_layer_loss = nn.CrossEntropyLoss()
        if not use_labeled_box:
            self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, targets=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        # encode context infomation
        add_losses = {}
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        '''flaggggggggggggggggggg'''
        pattern_pred = self.pattern_classifier1(prod_rep)
        if use_pattern_bias:
            pattern_pred = pattern_pred + self.freq_bias.index_with_labels_pattern(pair_pred.long())
        top_scores, top_class = pattern_pred[:, 1:].max(dim=1)

        if self.training:
            rel_labels = cat(rel_labels, dim=0)
            pattern_loss = self.multi_label_loss(pattern_pred, rel_labels.long())
            add_losses['pattern_loss'] = pattern_loss

        if self.training:
            if use_split:
                hier_visual, hier_prod, heri_label, pair_matrix, select_matrix = self.get_input_select(top_class,
                                                                                                       rel_labels,
                                                                                                       prod_rep,
                                                                                                       prod_rep,
                                                                                                       pair_pred)
            else:
                hier_visual, hier_prod, heri_label, pair_matrix, select_matrix = self.get_input(top_class, rel_labels,
                                                                                                prod_rep,
                                                                                                prod_rep, pair_pred)

            heri_output = self.get_heri_output(hier_visual, hier_prod, select_matrix, pair_matrix)
            self.count_second_layer_loss(heri_output, heri_label, add_losses, select_matrix)
            if bpr_loss:
                self.bpr_loss(heri_output, heri_label, add_losses, select_matrix)
            if not use_labeled_box:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
        else:
            if suppose_first_classify_is_right:
                rel_labels = cat(rel_labels, dim=0)
                hier_visual, hier_prod, pair_matrix, select_matrix, idx_sort = self.get_proper_input_for_test_fixed(
                    rel_labels, prod_rep, prod_rep, pair_pred)
            else:
                hier_visual, hier_prod, pair_matrix, select_matrix, idx_sort = self.get_proper_input_for_test(top_class,
                                                                                                              prod_rep,
                                                                                                              prod_rep,
                                                                                                              pair_pred)

            heri_output = self.get_heri_output(hier_visual, hier_prod, select_matrix, pair_matrix)
            rel_dists = self.test_result_cast(heri_output, len(top_class), select_matrix, idx_sort)

        if self.training:
            return None, None, add_losses
        else:
            obj_dists = obj_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)
            return obj_dists, rel_dists, add_losses

    def generate_one_hot_vector(self, myhier):
        dim2 = len(myhier) + 1
        onehot = np.zeros([51, dim2])
        for i in range(len(myhier)):
            for j in range(len(myhier[i])):
                data = myhier[i][j]
                onehot[data, i + 1] = 1.0
        onehot[0, 0] = 0.3
        return onehot

    def generate_muti_networks(self):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        # pattern classifier
        self.pattern_classifier1 = nn.Linear(self.pooling_dim, len(myhier) + 1)
        layer_init(self.pattern_classifier1, xavier=True)
        self.pattern_compress1 = nn.Linear(self.hidden_dim * 2, len(myhier) + 1)
        layer_init(self.pattern_compress1, xavier=True)

        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3]

        return classifer_all

    def get_input(self, top_class, rel_label, visual_rep, prod_rep, pair_pred):
        num_rel = len(myhier)
        visual_matrix = []
        prod_matrix = []
        target_label = []
        select_matrix = []
        pair_matrix = []
        for i in range(num_rel):
            visual_matrix.append([])
            prod_matrix.append([])
            target_label.append([])
            select_matrix.append(0)
            pair_matrix.append([])
        for i in range(len(rel_label)):
            tar = int(rel_label[i].item())
            piece_prod = prod_rep[i].unsqueeze(dim=0)
            piece_visual = visual_rep[i].unsqueeze(dim=0)
            piece_pair = pair_pred[i].unsqueeze(dim=0)
            if tar != 0:
                one_hot = self.onehot_for_51[tar]
                for j in range(len(one_hot)):
                    if one_hot[j] == 1.0:
                        idx = j - 1
                        prod_matrix[idx].append(piece_prod)
                        visual_matrix[idx].append(piece_visual)
                        pair_matrix[idx].append(piece_pair)
                        target_label[idx].append(rel_label[i])
            else:
                # idx = int(top_class[i].item())
                idx = i % 3
                if len(target_label[idx]) != 0:
                    prod_matrix[idx].append(piece_prod)
                    visual_matrix[idx].append(piece_visual)
                    pair_matrix[idx].append(piece_pair)
                    target_label[idx].append(0)

        ov = []
        om = []
        ol = []
        op = []

        for i in range(num_rel):
            if len(pair_matrix[i]) > 0:
                t2 = torch.tensor(target_label[i], dtype=rel_label.dtype, device=rel_label.device)
                if t2.sum().item() > 0:
                    select_matrix[i] = 1
                    t11 = cat(prod_matrix[i], dim=0)
                    om.append(t11)
                    t12 = cat(visual_matrix[i], dim=0)
                    ov.append(t12)
                    t2 = torch.tensor(target_label[i], dtype=rel_label.dtype, device=rel_label.device)
                    ol.append(t2)
                    t3 = cat(pair_matrix[i], dim=0)
                    op.append(t3)

        if len(om) == 0:
            for i in range(num_rel):
                k = num_rel-i-1
                if len(pair_matrix[k]) > 0:
                    t11 = cat(prod_matrix[k], dim=0)
                    om.append(t11)
                    t12 = cat(visual_matrix[k], dim=0)
                    ov.append(t12)
                    t2 = torch.tensor(target_label[k], dtype=rel_label.dtype, device=rel_label.device)
                    ol.append(t2)
                    t3 = cat(pair_matrix[k], dim=0)
                    op.append(t3)
                    select_matrix[k] = 1
                    break
        return ov, om, ol, op, select_matrix

    def get_input_select(self, top_class, rel_label, visual_rep, prod_rep, pair_pred):
        num_rel = len(myhier)
        visual_matrix = []
        prod_matrix = []
        target_label = []
        select_matrix = []
        pair_matrix = []
        for i in range(num_rel):
            visual_matrix.append([])
            prod_matrix.append([])
            target_label.append([])
            select_matrix.append(0)
            pair_matrix.append([])
        for i in range(len(rel_label)):
            tar = int(rel_label[i].item())
            piece_prod = prod_rep[i].unsqueeze(dim=0)
            piece_visual = visual_rep[i].unsqueeze(dim=0)
            piece_pair = pair_pred[i].unsqueeze(dim=0)
            if relabel_choice_mask[tar] == 0:
                idx = int(top_class[i].item())
                prod_matrix[idx].append(piece_prod)
                visual_matrix[idx].append(piece_visual)
                pair_matrix[idx].append(piece_pair)
                if tar in myhier[idx]:
                    target_label[idx].append(rel_label[i])
                else:
                    target_label[idx].append(0)

            elif relabel_choice_mask[tar] == 1:
                one_hot = self.onehot_for_51[tar]
                for j in range(len(one_hot)):
                    if one_hot[j] == 1.0:
                        idx = j - 1
                        prod_matrix[idx].append(piece_prod)
                        visual_matrix[idx].append(piece_visual)
                        pair_matrix[idx].append(piece_pair)
                        target_label[idx].append(rel_label[i])
            else:
                idx = int(top_class[i].item())
                if len(target_label[idx]) != 0:
                    prod_matrix[idx].append(piece_prod)
                    visual_matrix[idx].append(piece_visual)
                    pair_matrix[idx].append(piece_pair)
                    target_label[idx].append(0)

        ov = []
        om = []
        ol = []
        op = []

        for i in range(num_rel):
            if len(pair_matrix[i]) > 0:
                t2 = torch.tensor(target_label[i], dtype=rel_label.dtype, device=rel_label.device)
                if t2.sum().item() > 0:
                    select_matrix[i] = 1
                    t11 = cat(prod_matrix[i], dim=0)
                    om.append(t11)
                    t12 = cat(visual_matrix[i], dim=0)
                    ov.append(t12)
                    t2 = torch.tensor(target_label[i], dtype=rel_label.dtype, device=rel_label.device)
                    ol.append(t2)
                    t3 = cat(pair_matrix[i], dim=0)
                    op.append(t3)

        if len(om) == 0:
            for i in range(num_rel):
                k = num_rel-i-1
                if len(pair_matrix[k]) > 0:
                    t11 = cat(prod_matrix[k], dim=0)
                    om.append(t11)
                    t12 = cat(visual_matrix[k], dim=0)
                    ov.append(t12)
                    t2 = torch.tensor(target_label[k], dtype=rel_label.dtype, device=rel_label.device)
                    ol.append(t2)
                    t3 = cat(pair_matrix[k], dim=0)
                    op.append(t3)
                    select_matrix[k] = 1
                    break
        return ov, om, ol, op, select_matrix

    def get_proper_input_for_test(self, class_label, visual_rep, prod_rep, pair_pred):
        '''arrange the input and label by top_class,
        notice the sum of all the lists is num_rel'''
        num_rel = len(myhier)
        visual_matrix = []
        prod_matrix = []
        select_matrix = []
        idx_sort = []
        pair_matrix = []
        for i in range(num_rel):
            visual_matrix.append([])
            prod_matrix.append([])
            idx_sort.append([])
            select_matrix.append(0)
            pair_matrix.append([])
        for i in range(len(class_label)):
            c1 = int(class_label[i].item())
            piece_visual_rep = visual_rep[i].unsqueeze(dim=0)
            piece_prod_rep = prod_rep[i].unsqueeze(dim=0)
            visual_matrix[c1].append(piece_visual_rep)
            prod_matrix[c1].append(piece_prod_rep)
            idx_sort[c1].append(i)
            piece_pair = pair_pred[i].unsqueeze(dim=0)
            pair_matrix[c1].append(piece_pair)

        om = []
        on = []
        op = []

        for i in range(num_rel):
            if len(visual_matrix[i])>0:
                select_matrix[i] = 1
                t1 = cat(visual_matrix[i], dim=0)
                om.append(t1)
                t2 = cat(prod_matrix[i], dim=0)
                on.append(t2)
                p1 = cat(pair_matrix[i], dim=0)
                op.append(p1)

        return om, on, op, select_matrix, idx_sort

    def get_proper_input_for_test_fixed(self, rel_label, visual_rep, prod_rep, pair_pred):
        '''arrange the input and label by top_class,
        notice the sum of all the lists is num_rel
        warning: only for test!!!'''
        num_rel = len(myhier)
        visual_matrix = []
        prod_matrix = []
        idx_sort = []
        select_matrix = []
        pair_matirx = []
        for i in range(num_rel):
            visual_matrix.append([])
            prod_matrix.append([])
            idx_sort.append([])
            select_matrix.append(0)
            pair_matirx.append([])
        for i in range(len(rel_label)):
            piece_visual_rep = visual_rep[i].unsqueeze(dim=0)
            piece_prod_rep = prod_rep[i].unsqueeze(dim=0)
            piece_pair = pair_pred[i].unsqueeze(dim=0)
            tar = rel_label[i].item()
            if tar != 0:
                lvec = label_51_to_3_overlap[tar]
                if len(lvec) > 1:
                    ridx = random.randint(0, len(lvec) - 1)
                    c0 = lvec[ridx] - 1
                else:
                    c0 = lvec[0] - 1
                idx_sort[c0].append(i)
                visual_matrix[c0].append(piece_visual_rep)
                prod_matrix[c0].append(piece_prod_rep)
                pair_matirx[c0].append(piece_pair)
            else:
                ridx = random.randint(0, 2)
                idx_sort[ridx].append(i)
                visual_matrix[ridx].append(piece_visual_rep)
                prod_matrix[ridx].append(piece_prod_rep)
                pair_matirx[ridx].append(piece_pair)

        om = []
        on = []
        op = []

        for i in range(num_rel):
            if len(visual_matrix[i]) > 0:
                select_matrix[i] = 1
                t1 = cat(visual_matrix[i], dim=0)
                om.append(t1)
                t2 = cat(prod_matrix[i], dim=0)
                on.append(t2)
                p1 = cat(pair_matirx[i], dim=0)
                op.append(p1)

        return om, on, op, select_matrix, idx_sort

    def get_heri_output(self, hier_visual, hier_prob, select_matrix, pair_matrix):
        '''generate heri_output by select_matrix'''
        heri_rel_list = []
        j = 0
        for i in range(len(self.rel_classifer_all)):
            if select_matrix[i] == 1:
                classifer = self.rel_classifer_all[i]
                rel_result = classifer(hier_visual[j])

                if self.use_bias:
                    pair_pred = pair_matrix[j]
                    rel_result = rel_result + self.freq_bias.index_with_labels(pair_pred.long())
                heri_rel_list.append(rel_result)
                j = j+1

        return heri_rel_list

    def count_second_layer_loss(self, heri_score, heri_label, add_losses, select_matrix):
        '''count hier-classifer loss'''
        j = 0
        for i in range(len(myhier)):
            if select_matrix[i] == 1:
                loss_counter = self.second_layer_loss
                add_losses['%d_heri_loss'%(i+1)] = loss_counter(heri_score[j],heri_label[j].long())
                j = j + 1

    def bpr_loss(self, heri_score, heri_label, add_losses, select_matrix):
        count = 0
        head_idx_ = [20, 29, 31]
        for i in range(len(myhier)):
            if select_matrix[i] == 1:
                bpr_loss = 0.0
                for j in range(len(heri_label[count])):
                    if heri_label[count][j] != head_idx_[count] and heri_label[count][j] != 0:
                        bpr_loss = bpr_loss - (bpr_mask[i][heri_label[count][j]] * torch.log(torch.sigmoid(
                            heri_score[count][j][heri_label[count][j]] - heri_score[count][j][head_idx_[i]])))
                add_losses['%d_bpr_loss' % (i + 1)] = bpr_loss_factor * bpr_loss
                count = count + 1

    def test_result_cast(self, heri_output, num_rel, select_matrix, idx_sort):
        '''cast the result, arrange the output into a [num_rel, 51] matrix'''
        zero_mat = []
        self.list_51 = []
        for i in range(51):
            self.list_51.append(-10.)
        for i in range(num_rel):
            zero_mat.append(self.list_51)
        heri_51 = torch.tensor(zero_mat, dtype=heri_output[0].dtype, device=heri_output[0].device)

        cur = []
        for i in range(len(select_matrix)):
            if select_matrix[i] == 1:
                cur.append(i)

        for i in range(len(heri_output)):
            '''i represent the nonzero output idx'''
            '''c represent the absolute idx[0-5]'''
            c = cur[i]
            min_score, _ = heri_output[i].min(dim=1)
            for j in range(len(heri_output[i])):
                '''j represent the output[i] label_idx'''
                idx = idx_sort[c][j]
                heri_51[idx] = heri_output[i][j]
                # for k in range(len(myhier[i])):
                #     heri_51[idx][myhier[i][k]] = heri_output[i][j][myhier[i][k]]

        return heri_51



@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # if self.training:
        #     rel_labels = cat(rel_labels, dim=0)
        #     self.bpr_loss(rel_dists, rel_labels, add_losses)

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


    def bpr_loss(self, rel_dists, rel_labels, add_losses):
        head_idx = [20, 29, 31]
        bpr_loss = 0.0
        rel_dists = cat(rel_dists, dim=0)
        for ii in range(len(rel_labels)):
            for i in range(len(predicate_include_num)):
                for j in range(len(predicate_include_num[i])):
                    if (predicate_include_num[i][j] == rel_labels[ii].item()) and (rel_labels[ii].item() != 0):
                        bpr_loss = bpr_loss - torch.log(torch.sigmoid(rel_dists[ii][predicate_include_num[i][j]] - rel_dists[ii][head_idx[i]]))
                        bpr_loss = torch.tensor(bpr_loss, dtype=rel_dists.dtype, device=rel_dists.device)
        add_losses['bpr_loss'] = bpr_loss_factor * bpr_loss


@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, iteration, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True),])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)

        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.pooling_dim),
                                            nn.ReLU(inplace=True)
                                        ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))


    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append( head_rep[pair_idx[:,0]] - tail_rep[pair_idx[:,1]] )
            else:
                ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list



    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats

        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':   # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE': # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)

        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    import time
    print('\n\nthe model we use is %s\n'%cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR)
    if use_gt_box:
        if use_labeled_box:
            print('the task is predcls!\n\n')
        else:
            print('the task is sgcls!\n\n')
    else:
        assert use_labeled_box == False
        print('the task is sgdet!\n\n')
    assert use_gt_box == cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
    assert use_labeled_box == cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
    time.sleep(2)
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg,  in_channels)
