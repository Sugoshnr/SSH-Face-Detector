from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import traceback
from keras import backend as K
import config
from simple_parser import get_data
import data_generators
import ssh_model as model
import losses as losses
import roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, Flatten, Reshape
from keras.models import Model
import numpy as np
import cv2

sys.setrecursionlimit(40000)

C = config.Config()

def getLosses(clss, regr, img_data, C, module):
	R = roi_helpers.rpn_to_roi(clss, regr, C, K.image_dim_ordering(), module, use_regr=True, overlap_thresh=0.5, max_boxes=300)
	# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
	X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping, module)
	print(module)
	print (R)
	# print(X2, Y1, Y2, IouS)

	x_img = cv2.imread(img_data['filepath'])
	(width, height) = (img_data['width'], img_data['height'])
	(rows, cols, _) = x_img.shape
	(resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)
	x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
	final_image = np.zeros((C.im_size, C.im_size, 3))
	final_image[:resized_height, :resized_width, :] = x_img
	x_img = final_image
	

	if X2 is None:
		return -1, -1	
	neg_samples = np.where(Y1[0, :, -1] == 1)
	pos_samples = np.where(Y1[0, :, -1] == 0)
	# cv2.rectangle(img, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 255, 0), 1)

	if len(pos_samples) > 0:
		pos_samples = pos_samples[0]
	else:
		pos_samples = []

	return len(pos_samples), pos_samples




all_imgs, classes_count, class_mapping = get_data('')

if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))


config_output_filename = 'config_options'

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

# train_imgs = [s for s in all_imgs if (s['imageset'] == 'train' and '998.jpg' in s['filepath'])]
train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
val_imgs = [s for s in all_imgs if s['imageset'] == 'val']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []

input_shape_img = (C.im_size, C.im_size, 3)

num_anchors = len(C.anchor_box_scales['M1']) * len(C.anchor_box_ratios)

img_input = Input(shape=input_shape_img)

SSH = model.getModel()
optimizer = Adam(lr=1e-5)
SSH.summary()
SSH.load_weights('generator_model_ssh.hdf5')
SSH.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), \
										losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), \
										losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

epoch_length = 1000
num_epochs = int(5000)
iter_num = 0

# validation_steps = 50

losses = np.zeros((epoch_length, 5))
losses_val = np.zeros((epoch_length, 5))

rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

best_val_loss = np.Inf
patience = 5
batchSize = 4

vis = True

# print(SSH.get_weights())

X, Y, img_data = next(data_gen_train)
P_rpn = SSH.predict_on_batch(X)

pos_samples_len_M1, pos_samples_M1 = getLosses(P_rpn[0], P_rpn[1], img_data, C, 'M1')
pos_samples_len_M2, pos_samples_M2 = getLosses(P_rpn[2], P_rpn[3], img_data, C, 'M2')
pos_samples_len_M3, pos_samples_M3 = getLosses(P_rpn[4], P_rpn[5], img_data, C, 'M3')
if pos_samples_M1 == -1 and pos_samples_M2 == -1 and pos_samples_M3 == -1:
	rpn_accuracy_rpn_monitor.append(0)
	rpn_accuracy_for_epoch.append(0)
	# continue


# print (pos_samples_M1)
# print (pos_samples_M2)
# print (pos_samples_M3)


# checkpoint = ModelCheckpoint('generator_'+C.model_path, monitor = 'val_loss', save_best_only = True, verbose = 1)
# earlystop = EarlyStopping(monitor = 'val_loss', patience = patience, verbose = 1)

# SSH.fit_generator(generator = data_gen_train,
# 				steps_per_epoch = len(train_imgs)//batchSize,
# 				validation_data = data_gen_val,
# 				validation_steps = len(val_imgs)//batchSize,
# 				epochs = num_epochs,
# 				callbacks = [checkpoint, earlystop])




