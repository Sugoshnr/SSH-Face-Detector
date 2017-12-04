from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import ssh_model as model
import roi_helpers
from simple_parser import get_data
import data_generators
import losses
from keras.optimizers import Adam, SGD, RMSprop
import json

sys.setrecursionlimit(40000)

config_output_filename = 'config_options'

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

# img_path = img_data['filepath']
img_path = 'test'


def LoadWeights(model):
	with open('weights/mahyarnajibi_ssh_weights.pickle', 'rb') as f:
		weights = pickle.load(f, encoding='latin1')
		new_weights = dict()
		for name, w in weights.items():
			if "@" in name:
				name = name.replace("@", "_")
			new_weights[name] = w
		weights = new_weights
	for layer in model.layers:
		# if weights.has_key(layer.name):
		if layer.name in weights:
			w, b = weights[layer.name]
			#print w.shape, b.shape, len(layer.name)-5
		#if layer.name[len(layer.name)-5:] == 'score':
	## TODO: Check transpose axes orderings, specifically 3,2,1,0
			w1 = np.transpose(w, (2,3,1,0))
			if '_score' in layer.name:
				# w1 = w1[:,:,:,0:4:2]
				# b = b[0:4:2]
				w1 = w1[:,:,:,1:4:2]
				b = b[1:4:2]
			#w1 = w1[:,:,:,0:4::2]
			print (w1.shape, b.shape)
			model.get_layer(layer.name).set_weights([w1,b])
			print ("Copied weights for layer: ", layer.name)
		else:
			print ("Warning!!! No weights found for: ", layer.name)
	return model
# def format_img_size(img, C):
# 	""" formats the image size based on config """
# 	img_min_side = float(C.im_size)
# 	(height,width,_) = img.shape
		
# 	if width <= height:
# 		ratio = img_min_side/width
# 		new_height = int(ratio * height)
# 		new_width = int(img_min_side)
# 	else:
# 		ratio = img_min_side/height
# 		new_width = int(ratio * width)
# 		new_height = int(img_min_side)
# 	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
# 	return img, ratio	


def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	if width >= height:
		ratio = width / img_min_side
		new_height = int(height / ratio)
		new_width = int(img_min_side)
	else:
		ratio = height / img_min_side
		new_width = int(width / ratio)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	# final_image = np.zeros((C.im_size, C.im_size, 3))
	# final_image[:new_height, :new_width, :] = img
	# img = final_image
	return img, ratio


def drawRect(R, prob, module, color, img):
		cnt=0
		for rect in R:
			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(
				ratio, 
				C.rpn_stride[module]*rect[0], 
				C.rpn_stride[module]*rect[1], 
				C.rpn_stride[module]*rect[2], 
				C.rpn_stride[module]*rect[3])
			cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), color, 1)
			# cv2.putText(img, str(prob_M1[cnt]), (real_x1, real_y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
			cv2.imwrite('test/out/{}_{}_{}.png'.format(idx, module, 'pretrained_generator_1'),img)
			cnt+=1
	


def ROI(Y1, Y2, C, K, module):
	# R, prob = roi_helpers.rpn_to_roi(Y1, Y2, C, K, module, overlap_thresh=0.5)
	box, prob = roi_helpers.rpn_to_roi(Y1, Y2, C, K, module, overlap_thresh=0.1, max_boxes=20)
	modules = np.zeros(prob.shape)
	modules[:] = int(module[-1])
	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	# R[:, 2] -= R[:, 0]
	# R[:, 3] -= R[:, 1]
	# return R, prob
	# print(R)
	return box, prob, modules


def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
# def get_real_coordinates(ratio, x1, y1, x2, y2):

# 	real_x1 = int(round(x1 // ratio))
# 	real_y1 = int(round(y1 // ratio))
# 	real_x2 = int(round(x2 // ratio))
# 	real_y2 = int(round(y2 // ratio))

# 	return (real_x1, real_y1, real_x2 ,real_y2)

def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = (np.round(x1 * ratio)).astype('i')
	real_y1 = (np.round(y1 * ratio)).astype('i')
	real_x2 = (np.round(x2 * ratio)).astype('i')
	real_y2 = (np.round(y2 * ratio)).astype('i')
	# real_x1 = int(round(x1 * ratio))
	# real_y1 = int(round(y1 * ratio))
	# real_x2 = int(round(x2 * ratio))
	# real_y2 = int(round(y2 * ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
# C.num_rois = int(options.num_rois)

# if C.network == 'resnet50':
# 	num_features = 1024
# elif C.network == 'vgg':
# 	num_features = 512

# if K.image_dim_ordering() == 'th':
# 	input_shape_img = (3, None, None)
# 	input_shape_features = (num_features, None, None)
# else:
# 	input_shape_img = (None, None, 3)
# 	input_shape_features = (None, None, num_features)


# img_input = Input(shape=input_shape_img)
# roi_input = Input(shape=(C.num_rois, 4))
# feature_map_input = Input(shape=input_shape_features)

# # define the base network (resnet here, can be VGG, Inception, etc)
# shared_layers = nn.nn_base(img_input, trainable=True)

# # define the RPN, built on the base layers
# num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
# rpn_layers = nn.rpn(shared_layers, num_anchors)

# classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

# model_rpn = Model(img_input, rpn_layers)
# model_classifier_only = Model([feature_map_input, roi_input], classifier)

# model_classifier = Model([feature_map_input, roi_input], classifier)

# print('Loading weights from {}'.format(C.model_path))
# model_rpn.load_weights(C.model_path, by_name=True)
# model_classifier.load_weights(C.model_path, by_name=True)

# model_rpn.compile(optimizer='sgd', loss='mse')
# model_classifier.compile(optimizer='sgd', loss='mse')
num_anchors = len(C.anchor_box_scales['M1']) * len(C.anchor_box_ratios)
SSH = model.getModel()
optimizer = Adam(lr=1e-5)
SSH.summary()
SSH.load_weights('weights/pretrained_generator_model_ssh_2.hdf5', by_name=True)
# SSH = LoadWeights(SSH)
# print(SSH.get_weights())

SSH.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), \
										losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), \
										losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])

all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True



for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	print(img_name)
	st = time.time()
	filepath = os.path.join(img_path,img_name)

	img = cv2.imread(filepath)

	X, ratio = format_img(img, C)

	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	# [Y1, Y2, F] = SSH.predict(X)
	
	[M1_Y1, M1_Y2, M2_Y1, M2_Y2, M3_Y1, M3_Y2] = SSH.predict(X)

	box_M1, prob_M1, module_M1 = ROI(M1_Y1, M1_Y2, C, K.image_dim_ordering(), 'M1')
	box_M2, prob_M2, module_M2 = ROI(M2_Y1, M2_Y2, C, K.image_dim_ordering(), 'M2')
	box_M3, prob_M3, module_M3 = ROI(M3_Y1, M3_Y2, C, K.image_dim_ordering(), 'M3')
	all_boxes = np.concatenate([box_M1, box_M2, box_M3])
	(all_boxes[:,0], all_boxes[:,1], all_boxes[:,2], all_boxes[:,3]) = get_real_coordinates(ratio, all_boxes[:,0], all_boxes[:,1], all_boxes[:,2], all_boxes[:,3])
	all_probs = np.concatenate([prob_M1, prob_M2, prob_M3])
	all_modules = np.concatenate([module_M1, module_M2, module_M3])
	# print(box_M1.shape)
	# print(box_M2.shape)
	# print(box_M3.shape)
	# print(all_boxes.shape)
	# print(all_probs.shape)
	# print(all_modules.shape)
	R, probs, modules = roi_helpers.non_max_suppression_fast_module(all_boxes, all_probs, all_modules, overlap_thresh=0.1, max_boxes=20)
	# print(R.shape)
	# # R[:, 2] -= R[:, 0]
	# # R[:, 3] -= R[:, 1]
	# print(len(R))
	for i in range(len(R)):
		# print(modules[i])
		stride = 'M'+str(int(modules[i]))
		if(stride == 'M1'):
			color = [255, 0, 0]
		if(stride == 'M2'):
			color = [0, 255, 0]
		if(stride == 'M3'):
			color = [0, 0, 255]
		# print(stride)
		# (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, C.rpn_stride[stride]*R[i][0], C.rpn_stride[stride]*R[i][1], C.rpn_stride[stride]*R[i][2], C.rpn_stride[stride]*R[i][3])
		# (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, R[i][0], R[i][1], R[i][2], R[i][3])
		(real_x1, real_y1, real_x2, real_y2) = (R[i][0], R[i][1], R[i][2], R[i][3])
		cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), color, 1)

	# R_M1, prob_M1 = ROI(M1_Y1, M1_Y2, C, K.image_dim_ordering(), 'M1')
	# R_M2, prob_M2 = ROI(M2_Y1, M2_Y2, C, K.image_dim_ordering(), 'M2')
	# R_M3, prob_M3 = ROI(M3_Y1, M3_Y2, C, K.image_dim_ordering(), 'M3')
	# drawRect(R_M1, prob_M1, 'M1', [255, 0, 0], img)
	# drawRect(R_M2, prob_M2, 'M2', [0, 255, 0], img)
	# drawRect(R_M3, prob_M3, 'M3', [0, 0, 255], img)
	# for rect in R_M2:
	# 	(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, C.rpn_stride['M2']*rect[0], C.rpn_stride['M2']*rect[1], C.rpn_stride['M2']*rect[2], C.rpn_stride['M2']*rect[3])
	# 	cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), [255, 0, 0], 1)
	# 	cv2.imwrite('test/out/{}_{}_{}.png'.format(idx, 'M2', 'pretrained_generator_1'),img)
	# for rect in R_M3:
	# 	(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, C.rpn_stride['M3']*rect[0], C.rpn_stride['M3']*rect[1], C.rpn_stride['M3']*rect[2], C.rpn_stride['M3']*rect[3])
	# 	cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), [0, 0, 255], 1)
	# 	cv2.imwrite('test/out/{}_{}_{}.png'.format(idx, 'M3', 'pretrained_generator_1'),img)

	# Y1 = M1_Y1
	# Y2 = M1_Y2
	# R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), 'M1', overlap_thresh=0.7)

	# # convert from (x1,y1,x2,y2) to (x,y,w,h)
	# R[:, 2] -= R[:, 0]
	# R[:, 3] -= R[:, 1]

	# print(R)
	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	# for jk in range(R.shape[0]//C.num_rois + 1):
	# 	ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
	# 	if ROIs.shape[1] == 0:
	# 		break

	# 	if jk == R.shape[0]//C.num_rois:
	# 		#pad R
	# 		curr_shape = ROIs.shape
	# 		target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
	# 		ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
	# 		ROIs_padded[:, :curr_shape[1], :] = ROIs
	# 		ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
	# 		ROIs = ROIs_padded
	# 	print (ROIs)
# 		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

# 		for ii in range(P_cls.shape[1]):

# 			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
# 				continue

# 			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

# 			if cls_name not in bboxes:
# 				bboxes[cls_name] = []
# 				probs[cls_name] = []

# 			(x, y, w, h) = ROIs[0, ii, :]

# 			cls_num = np.argmax(P_cls[0, ii, :])
# 			try:
# 				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
# 				tx /= C.classifier_regr_std[0]
# 				ty /= C.classifier_regr_std[1]
# 				tw /= C.classifier_regr_std[2]
# 				th /= C.classifier_regr_std[3]
# 				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
# 			except:
# 				pass
# 			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
# 			probs[cls_name].append(np.max(P_cls[0, ii, :]))

# 	all_dets = []

# 	for key in bboxes:
# 		bbox = np.array(bboxes[key])

# 		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
# 		for jk in range(new_boxes.shape[0]):
# 			(x1, y1, x2, y2) = new_boxes[jk,:]

# 			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

# 			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

# 			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
# 			all_dets.append((key,100*new_probs[jk]))

# 			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
# 			textOrg = (real_x1, real_y1-0)

# 			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
# 			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
# 			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

# 	print('Elapsed time = {}'.format(time.time() - st))
# 	print(all_dets)
	# cv2.imshow('img', img)
	# cv2.waitKey(0)
	cv2.imwrite('test/out/pretrained_1/{}_1.png'.format(idx),img)
	# break;
