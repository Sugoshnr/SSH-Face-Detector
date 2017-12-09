import config
from simple_parser import get_data
import data_generators
import numpy as np
from keras import backend as K
import os
import pickle
import data_generators_2 as data_generators
from sys import getsizeof
import time
import sys
import roi_helpers

C = config.Config()

# all_imgs, classes_count, class_mapping = get_data('')
# train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
# val_imgs = [s for s in all_imgs if s['imageset'] == 'val']

# print('Num train samples {}'.format(len(train_imgs)))
# print('Num val samples {}'.format(len(val_imgs)))

# data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')
# data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

# cnt = 0
# while cnt < len(train_imgs):
# 	X, Y, img_data = next(data_gen_train)
# 	p = img_data['filepath'].split(os.sep)[:-1]
# 	f = img_data['filepath'].split(os.sep)[-1]+'.pickle'
# 	final_path = (os.path.join(os.sep.join(p), f))
# 	# print(p, f, final_path)
# 	# print(img_data.keys())
# 	# print((img_data[img_data['stride']]))
# 	# print(getsizeof(img_data))
# 	# # A = np.asarray(img_data)
# 	# # np_compression.compress(A)
# 	pickle.dump(img_data, open(final_path, "wb" ))
# 	cnt+=1
# 	print(cnt)
	
# cnt = 0
# while cnt < len(val_imgs):
# 	X, Y, img_data = next(data_gen_train)
# 	p = img_data['filepath'].split(os.sep)[:-1]
# 	f = img_data['filepath'].split(os.sep)[-1]+'.pickle'
# 	final_path = (os.path.join(os.sep.join(p), f))
# 	# print(p, f, final_path)
# 	# print(img_data.keys())
# 	# print((img_data[img_data['stride']]))
# 	# print(getsizeof(img_data))
# 	# # A = np.asarray(img_data)
# 	# # np_compression.compress(A)
# 	pickle.dump(img_data, open(final_path, "wb" ))
# 	cnt+=1
# 	print(cnt)
	

# img_data = (pickle.load(open('test.pickle', 'rb')))
# # print(img_data)
# print('TEST-----')
# print(img_data['stride'])
# stride = img_data['stride']
# y_is_box_valid, y_rpn_overlap, y_rpn_regr = img_data[img_data['stride']]['M1']
# pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
# neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
# neutral_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 0))

# print('{} {} pos, neg, neut = {} {} {}'.format('M1', stride, len(pos_locs[0]), len(neg_locs[0]), len(neutral_locs[0])))
# print(y_is_box_valid.shape, y_rpn_overlap.shape, y_rpn_regr.shape)

# y_is_box_valid, y_rpn_overlap, y_rpn_regr = img_data[img_data['stride']]['M2']
# pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
# neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
# neutral_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 0))

# print('{} {} pos, neg, neut = {} {} {}'.format('M2', stride, len(pos_locs[0]), len(neg_locs[0]), len(neutral_locs[0])))
# print(y_is_box_valid.shape, y_rpn_overlap.shape, y_rpn_regr.shape)

# y_is_box_valid, y_rpn_overlap, y_rpn_regr = img_data[img_data['stride']]['M3']
# pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
# neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
# neutral_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 0))

# print('{} {} pos, neg, neut = {} {} {}'.format('M3', stride, len(pos_locs[0]), len(neg_locs[0]), len(neutral_locs[0])))
# print(y_is_box_valid.shape, y_rpn_overlap.shape, y_rpn_regr.shape)


# print (repr['M2'])
# print (repr['M3'])
# pickle.dump(C.representation, open( "WIDER_target_anchors.pickle", "wb" ))
