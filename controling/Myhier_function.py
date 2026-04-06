import numpy as np
import math

predicate_lists = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
predicate_dict = {'__background__': 0, 'above': 1, 'across': 2, 'against': 3, 'along': 4, 'and': 5, 'at': 6, 'attached to': 7, 'behind': 8, 'belonging to': 9, 'between': 10, 'carrying': 11, 'covered in': 12, 'covering': 13, 'eating': 14, 'flying in': 15, 'for': 16, 'from': 17, 'growing on': 18, 'hanging from': 19, 'has': 20, 'holding': 21, 'in': 22, 'in front of': 23, 'laying on': 24, 'looking at': 25, 'lying on': 26, 'made of': 27, 'mounted on': 28, 'near': 29, 'of': 30, 'on': 31, 'on back of': 32, 'over': 33, 'painted on': 34, 'parked on': 35, 'part of': 36, 'playing': 37, 'riding': 38, 'says': 39, 'sitting on': 40, 'standing on': 41, 'to': 42, 'under': 43, 'using': 44, 'walking in': 45, 'walking on': 46, 'watching': 47, 'wearing': 48, 'wears': 49, 'with': 50}
class_count = [0, 8411, 263, 224, 493, 679, 2109, 1586, 13047, 652, 511, 1705, 485, 512, 702, 5, 1116, 198, 172, 807, 69007, 11482, 24470, 3808, 779, 1026, 369, 128, 265, 20759, 32770, 118037, 343, 1277, 153, 641, 435, 134, 4507, 49, 5355, 2496, 327, 4732, 580, 289, 1322, 907, 48582, 4939, 12215]
data_count = {'above': 11602, 'across': 364, 'against': 318, 'along': 683, 'and': 933, 'at': 3030, 'attached to': 2689, 'behind': 18159, 'belonging to': 1518, 'between': 765, 'carrying': 2504, 'covered in': 705, 'covering': 828, 'eating': 1063, 'flying in': 30, 'for': 1507, 'from': 306, 'growing on': 296, 'hanging from': 1296, 'has': 97473, 'holding': 16514, 'in': 37914, 'in front of': 5609, 'laying on': 1138, 'looking at': 1410, 'lying on': 561, 'made of': 172, 'mounted on': 486, 'near': 30331, 'of': 53761, 'on': 196495, 'on back of': 547, 'over': 1871, 'painted on': 263, 'parked on': 1162, 'part of': 624, 'playing': 166, 'riding': 6335, 'says': 62, 'sitting on': 8003, 'standing on': 4068, 'to': 519, 'under': 6687, 'using': 822, 'walking in': 426, 'walking on': 2343, 'watching': 1327, 'wearing': 71121, 'wears': 7471, 'with': 18428}
data_count_sorted = [('on', 196495), ('has', 97473), ('wearing', 71121), ('of', 53761), ('in', 37914), ('near', 30331), ('with', 18428), ('behind', 18159), ('holding', 16514), ('above', 11602),
                     ('sitting on', 8003), ('wears', 7471), ('under', 6687), ('riding', 6335), ('in front of', 5609), ('standing on', 4068), ('at', 3030), ('attached to', 2689), ('carrying', 2504), ('walking on', 2343), ('over', 1871), ('belonging to', 1518), ('for', 1507), ('looking at', 1410), ('watching', 1327), ('hanging from', 1296), ('parked on', 1162), ('laying on', 1138), ('eating', 1063),
                     ('and', 933), ('covering', 828), ('using', 822), ('between', 765), ('covered in', 705), ('along', 683), ('part of', 624), ('lying on', 561), ('on back of', 547), ('to', 519), ('mounted on', 486), ('walking in', 426), ('across', 364), ('against', 318), ('from', 306), ('growing on', 296), ('painted on', 263), ('made of', 172), ('playing', 166), ('says', 62), ('flying in', 30)]


class_num = [200000, 11602, 364, 318, 683, 933, 3030, 2689, 18159, 1518, 765, 2504, 705, 828, 1063, 30, 1507, 306, 296, 1296, 97473, 16514, 37914, 5609, 1138, 1410, 561, 172, 486, 30331, 53761, 196495, 547, 1871, 263, 1162, 624, 166, 6335, 62, 8003, 4068, 519, 6687, 822, 426, 2343, 1327, 71121, 7471, 18428]

predicate_include_added_num = [[5, 12, 14, 27, 37, 39, 44, 50],
                                 [3, 4, 5, 10, 23, 27],
                                 [1, 2, 3, 4, 5, 7, 10, 12, 13, 15, 17, 18, 19, 24, 26, 27, 28, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46]]
predicate_include_num = [[5, 12, 14, 27, 37, 39, 44],
                        [3, 4, 5, 10, 27],
                        [2, 3, 4, 5, 10, 12, 13, 15, 17, 18, 19, 24, 26, 27, 28, 32, 34, 35, 36, 37, 39, 42, 44, 45]]

predicate_include_num_2k = [[5, 12, 14, 16, 25, 27, 37, 39, 44],
                            [3, 4, 5, 10, 16, 25, 27, 47],
                            [2, 3, 4, 5, 9, 10, 12, 13, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 39, 42, 44, 45, 47]]

def generate_list_for_next(predicate_graph):
    label_51_to_6 = []
    label_new_allocate = []
    for i in range(51):
        label_51_to_6.append([])
        label_new_allocate.append({})
    label_51_to_6[0].append(0)
    label_new_allocate[0][0] = 0
    for i in range(len(predicate_graph)):
        for j in range(len(predicate_graph[i])):
            label_51_to_6[predicate_graph[i][j]].append(i+1)
            label_new_allocate[predicate_graph[i][j]][i+1] = j+1

    bpr_mask = []
    for i in range(len(predicate_graph)):
        bpr_mask.append([])
        bpr_mask[i].append(0)
        for j in range(1, 51):
            if j in predicate_include_num[i]:
                bpr_mask[i].append(1)
            else:
                bpr_mask[i].append(0)
    return label_51_to_6, label_new_allocate, bpr_mask

def get_weight(myhier, beta):
    effective_num = 1.0 - np.power(beta, class_num)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights_ = per_cls_weights / np.sum(per_cls_weights) * len(class_num)
    return per_cls_weights_

    # hier_num = []
    # for i in range(len(myhier)):
    #     hier_num.append([])
    #     for j in range(len(myhier[i])):
    #         hier_num[i].append(class_num[myhier[i][j]])
    # max = []
    # for i in range(len(hier_num)):
    #     max.append(np.max(hier_num[i]))
    #
    # weight = []
    #
    # for i in range(len(hier_num)):
    #     hier_num51 = []
    #     for j in range(51):
    #         if j in myhier[i]:
    #             hier_num51.append(class_num[j])
    #         else:
    #             hier_num51.append(max[i])
    #     effective_num = 1.0 - np.power(beta, hier_num51)
    #     per_cls_weights = (1.0 - beta) / np.array(effective_num)
    #     per_cls_weights_ = per_cls_weights / np.sum(per_cls_weights) * 51
    #     weight.append(list(per_cls_weights_))

    # return weight