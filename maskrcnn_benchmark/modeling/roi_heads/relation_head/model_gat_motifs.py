# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, center_x, sort_by_score, to_onehot, get_dropout_mask, nms_overlaps, encode_box_info
from torch_geometric.nn import GATConv

class GATEdgeNet_M(torch.nn.Module):
    def __init__(self, config, in_channels):
        super(GATEdgeNet_M, self).__init__()
        self.cfg = config
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.obj_dim = in_channels
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.encoder_size = int(self.hidden_dim/self.num_head)
        # print(dataset.num_features) #1433
        self.conv1 = GATConv(self.embed_dim + self.hidden_dim + self.obj_dim, self.encoder_size, heads=self.num_head, dropout=0.1)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(self.hidden_dim, self.hidden_dim, heads=1, concat=False, dropout=0.1)

    def forward(self, edge_feature, edge_index):
        # x = F.dropout(data.x, p=0.6, training=self.training)
        edge_f1 = F.elu(self.conv1(edge_feature, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        edge_output = self.conv2(edge_f1, edge_index)
        return edge_output

class GATNet_Object(torch.nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(GATNet_Object, self).__init__()
        self.cfg = config
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels

        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.obj_dim = in_channels
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.encoder_size = int(self.hidden_dim/self.num_head)

        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])

        self.out_obj = nn.Linear(self.hidden_dim, len(self.obj_classes))

        self.lin_obj = nn.Linear(self.in_channels + self.embed_dim + 128, self.hidden_dim)

        # print(dataset.num_features) #1433
        self.conv1 = GATConv(self.embed_dim + 128 + self.obj_dim, self.encoder_size, heads=self.num_head, dropout=0.1)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(self.hidden_dim, self.hidden_dim, heads=1, concat=False, dropout=0.1)

    def forward(self, roi_features, proposals, edge_index, logger=None):
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        # bbox embedding will be used as input
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # encode objects with transformer
        obj_pre_rep = cat((roi_features, obj_embed, pos_embed), -1)
        num_objs = [len(p) for p in proposals]

        ''''''
        # x = F.dropout(data.x, p=0.6, training=self.training)
        edge_f1 = F.elu(self.conv1(obj_pre_rep, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        obj_feats = self.conv2(edge_f1, edge_index)

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
            edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_labels)), dim=-1)
        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
            edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_preds)), dim=-1)

        return obj_dists, obj_preds, edge_pre_rep

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds
