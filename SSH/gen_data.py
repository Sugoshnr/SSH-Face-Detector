import config
from simple_parser import get_data
import data_generators
import numpy as np
from keras import backend as K
import os
import pickle

C = config.Config()

all_imgs, classes_count, class_mapping = get_data('')
train_imgs = [s for s in all_imgs if s['imageset'] == 'train' and '0_Parade_marchingband_1_381.jpg' in s['filepath']]
# val_imgs = [s for s in all_imgs if s['imageset'] == 'val']

print('Num train samples {}'.format(len(train_imgs)))
# print('Num val samples {}'.format(len(val_imgs)))

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')
# data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

# cnt = 0
# while cnt < 2:
X, Y, img_data = next(data_gen_train)
# 	p = img_data['filepath'].split(os.sep)[:-1]
# 	f = img_data['filepath'].split(os.sep)[-1]+'.pickle'
# 	final_path = (os.path.join(os.sep.join(p), f))
# 	print(p,f, final_path)
# 	pickle.dump(C.representation, open( final_path, "wb" ))
# 	cnt+=1
# 	print(cnt)
	
# cnt = 0
# while cnt < 2:
# 	X, Y, img_data = next(data_gen_val)
# 	p = img_data['filepath'].split(os.sep)[:-1]
# 	f = img_data['filepath'].split(os.sep)[-1]+'.pickle'
# 	final_path = (os.path.join(os.sep.join(p), f))
# 	print(p,f, final_path)
# 	pickle.dump(C.representation, open( final_path, "wb" ))
# 	cnt+=1
# 	print(cnt)
	

representation = (pickle.load(open('WIDER/WIDER_train/images/0--Parade/0_Parade_marchingband_1_381.jpg.pickle', 'rb'), encoding='latin1'))
y_is_box_valid, y_rpn_overlap, y_rpn_regr = representation['M1']
pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
neutral_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 0))

print('pos, neg, neut = ', len(pos_locs[0]), len(neg_locs[0]), len(neutral_locs[0]))

y_is_box_valid, y_rpn_overlap, y_rpn_regr = representation['M2']
pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
neutral_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 0))

print('pos, neg, neut = ', len(pos_locs[0]), len(neg_locs[0]), len(neutral_locs[0]))

y_is_box_valid, y_rpn_overlap, y_rpn_regr = representation['M3']
pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
neutral_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 0))

print('pos, neg, neut = ', len(pos_locs[0]), len(neg_locs[0]), len(neutral_locs[0]))


# print (repr['M2'])
# print (repr['M3'])
# pickle.dump(C.representation, open( "WIDER_target_anchors.pickle", "wb" ))
