import time
import numpy as np
from controling.Myhier_function import generate_list_for_next, get_weight


##################################  hier训练时的参数设置   ####################################
use_bias = True                 # 1 是否使用bias
use_gt_box = True              # 是否使用gt box 【sgdet】
use_labeled_box = True         # 使用的gt box是否含label 【sgcls or predcls】

##################################  bpr Loss的参数设置   ####################################
bpr_loss = True                 # 3 是否使用bpr
bpr_loss_factor = 0.02         # 4 bpr系数

use_split = False            # 是否使用relabel_choice_mask

use_pattern_bias = False
use_weight = True
##################################  测试时的参数设置   ####################################
suppose_first_classify_is_right = False  # 8 是否进行上界分析（仅测试）

##################################  以下是预定义的hier信息   ####################################
head_pairs_names = ['__background__', 'has', 'near', 'on']
myhier = [[5, 11, 12, 14, 16, 20, 21, 22, 25, 27, 37, 39, 43, 44, 48, 49, 50],
          [3, 4, 5, 6, 8, 10, 16, 23, 25, 27, 29, 43, 47],
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]

relabel_choice_mask = [-1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
                       0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
weight = get_weight(myhier, beta=0.9999)
##################################  以下是生成辅助数组   ####################################
label_51_to_3_overlap, label_new_allocate_ol, bpr_mask = generate_list_for_next(myhier)

##################################  以下是打印辅助提示   ####################################


if False:
    print_message()

if suppose_first_classify_is_right:
    print('\nwe suppose the first classifer is absolutely right, to test ceiling!')
if use_bias:
    print('\nuse bias!')
    time.sleep(1)
else:
    print('\ndo not use bias!')
    time.sleep(1)
if bpr_loss:
    print('\nuse bpr loss!')
    print('bpr parameter is %.4f' % (bpr_loss_factor))
    time.sleep(1)

