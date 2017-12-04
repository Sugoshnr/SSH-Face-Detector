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

from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, Flatten, Reshape
from keras.models import Model
import numpy as np

sys.setrecursionlimit(40000)

C = config.Config()

def getLosses(clss, regr, img_data, C, module):
	R = roi_helpers.rpn_to_roi(clss, regr, C, K.image_dim_ordering(), module, use_regr=True, overlap_thresh=0.5, max_boxes=300)
	# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
	X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping, module)

	if X2 is None:
		return -1	
	neg_samples = np.where(Y1[0, :, -1] == 1)
	pos_samples = np.where(Y1[0, :, -1] == 0)

	if len(pos_samples) > 0:
		pos_samples = pos_samples[0]
	else:
		pos_samples = []

	return len(pos_samples)




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
SSH.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), \
										losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), \
										losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

epoch_length = 1000
num_epochs = int(5000)
iter_num = 0

validation_steps = 50

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

vis = True

for epoch_num in range(num_epochs):

	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

	while True:
		try:
			if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
				if mean_overlapping_bboxes == 0:
					print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

			X, Y, img_data = next(data_gen_train)

			loss_rpn = SSH.train_on_batch(X, Y)

			rpn_accuracy_rpn_monitor.append(0)
			rpn_accuracy_for_epoch.append(0)
		
			# P_rpn = SSH.predict_on_batch(X)
			
			# pos_samples_M1 = getLosses(P_rpn[0], P_rpn[1], img_data, C, 'M1')
			# pos_samples_M2 = getLosses(P_rpn[2], P_rpn[3], img_data, C, 'M2')
			# pos_samples_M3 = getLosses(P_rpn[4], P_rpn[5], img_data, C, 'M3')
			# if pos_samples_M1 == -1 and pos_samples_M2 == -1 and pos_samples_M3 == -1:
			# 	rpn_accuracy_rpn_monitor.append(0)
			# 	rpn_accuracy_for_epoch.append(0)
			# 	continue

			# rpn_accuracy_rpn_monitor.append(pos_samples_M1 + pos_samples_M2 + pos_samples_M3)
			# rpn_accuracy_for_epoch.append(pos_samples_M1 + pos_samples_M2 + pos_samples_M3)

			losses[iter_num, 0] = loss_rpn[1] + loss_rpn[3] + loss_rpn[5]
			losses[iter_num, 1] = loss_rpn[2] + loss_rpn[4] + loss_rpn[6]

			iter_num += 1

			progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1]))])
									  # ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])
			if iter_num == epoch_length:
			# if True:
				iter_num = 0
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])


				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				if C.verbose:
					print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					# print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Loss RPN regression: {}'.format(loss_rpn_regr))
					# print('Loss Detector classifier: {}'.format(loss_class_cls))
					# print('Loss Detector regression: {}'.format(loss_class_regr))
					print('Elapsed time: {}'.format(time.time() - start_time))

				curr_loss = loss_rpn_cls + loss_rpn_regr # + loss_class_cls + loss_class_regr

				if curr_loss < best_loss:
					if C.verbose:
						print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
					SSH.save_weights(C.model_path)

				print('Validation Started')
				val_iter = 0
				while(val_iter < validation_steps):
					val_iter+=1
					X_val, Y_val, img_data_val = next(data_gen_val)
					loss_rpn_val = SSH.test_on_batch(X_val, Y_val)
					losses_val[val_iter, 0] = loss_rpn_val[1] + loss_rpn_val[3] + loss_rpn_val[5]
					losses_val[val_iter, 1] = loss_rpn_val[2] + loss_rpn_val[4] + loss_rpn_val[6]
				loss_rpn_cls_val = np.mean(losses_val[:, 0])
				loss_rpn_regr_val = np.mean(losses_val[:, 1])
				curr_loss_val = loss_rpn_cls_val + loss_rpn_regr_val
				print('Validation loss = ', curr_loss_val)
				if(patience>=5 and curr_loss_val>best_val_loss):
					raise Exception('Validation Early Stopping')
				elif(curr_loss_val>best_val_loss and patience<5):
					patience+=1
				elif(curr_loss_val<best_val_loss):
					patience=0
					best_val_loss=curr_loss_val
				print('Validation Stopped')

				start_time = time.time()
				break

		except Exception as e:
			print('Exception: {}'.format(e))
			print(traceback.format_exc())
			continue


