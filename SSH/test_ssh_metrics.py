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
import random

sys.setrecursionlimit(40000)

C = config.Config()

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
	box, prob = roi_helpers.rpn_to_roi(Y1, Y2, C, K, module, overlap_thresh=0.3, max_boxes=1000)
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
	return (real_x1, real_y1, real_x2 ,real_y2)



def f_measure(gt_bboxes, pred_bboxes):
    deteval = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype='int')
    for gt_idx in range(deteval.shape[0]):
        for pred_idx in range(deteval.shape[1]):
            gt_bbox = [gt_bboxes[gt_idx]['x1'], gt_bboxes[gt_idx]['y1'], gt_bboxes[gt_idx]['x2'], gt_bboxes[gt_idx]['y2']]
            pred_bbox = pred_bboxes[pred_idx]
            curr_iou = data_generators.IoU(gt_bbox, pred_bbox)
            deteval[gt_idx, pred_idx] = 1 if curr_iou >= C.rpn_max_overlap else 0
    r_count = np.sum(np.sum(deteval, axis=1) > 0)
    r = r_count / float(len(gt_bboxes))
    p_count = np.sum(np.sum(deteval, axis=0) > 0)
    p = p_count / float(len(pred_bboxes))
    f = 0 if r == 0 or p == 0 else (2. * r * p) / (r + p)
    return r_count, r, p_count, p, f

def f_measure_rpn(C, rpn_bboxes, rpn_probs, gt_bboxes):
    # gt_bboxes = np.asarray([[bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
    #                         for bbox in C.img_data['bboxes_scaled_output']], dtype='int')
    # print(rpn_bboxes, gt_bboxes, rpn_probs)
    pred_bboxes = np.asarray([rpn_bboxes[idx, :] for idx in range(len(rpn_bboxes)) if rpn_probs[idx] >= 0.5])
    r_count, r, p_count, p, f = f_measure(gt_bboxes, pred_bboxes)
    print('''
          RPN recall: {} = {}/{},
          RPN precision: {} = {}/{},
          RPN f-measure: {}'''.format(r, r_count, len(gt_bboxes), p, p_count, len(pred_bboxes), f))
    C.eval['rpn_rcount'] = [r_count] if 'rpn_rcount' not in C.eval else C.eval['rpn_rcount'] + [r_count]
    C.eval['rpn_recall'] = [r] if 'rpn_recall' not in C.eval else C.eval['rpn_recall'] + [r]
    C.eval['rpn_pcount'] = [p_count] if 'rpn_pcount' not in C.eval else C.eval['rpn_pcount'] + [p_count]
    C.eval['rpn_precision'] = [p] if 'rpn_precision' not in C.eval else C.eval['rpn_precision'] + [p]
    C.eval['rpn_tcount'] = [len(gt_bboxes)] if 'rpn_tcount' not in C.eval else C.eval['rpn_tcount'] + [len(gt_bboxes)]
    C.eval['rpn_ocount'] = [len(pred_bboxes)] if 'rpn_ocount' not in C.eval else C.eval['rpn_ocount'] + [len(pred_bboxes)]
    # if C.diagnose:
    #     img = cv2.imread(C.img_data['filepath'])
    #     (width, height) = (C.img_data['width'], C.img_data['height'])
    #     (resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)
    #     img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    #     num_bboxes = len(C.img_data['bboxes'])
    #     for bbox_num in range(num_bboxes):
    #         bbox = C.img_data['bboxes_scaled_input'][bbox_num]
    #         x1_gt, x2_gt, y1_gt, y2_gt = map(int, [bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']])
    #         cv2.rectangle(img, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 255, 0), 1)
    #     num_bboxes = len(pred_bboxes)
    #     for bbox_num in range(num_bboxes):
    #         x1, y1, x2, y2 = map(lambda x: x * C.rpn_stride, pred_bboxes[bbox_num].astype('int'))
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
    #     p_ = C.img_data['filepath'].split(os.sep)[:-2]
    #     f_ = C.img_data['filepath'].split(os.sep)[-1]
    #     f_ = '_rpn.'.join(f_.split('.'))
    #     cv2.imwrite(os.path.join(os.sep.join(p_), 'diagnose', f_), img)
    #     # cv2.imshow('gt boxes and anchors {}'.format(f), img)
    #     # cv2.waitKey(0)
    return f, r, p


all_imgs, classes_count, class_mapping = get_data('')
random.shuffle(all_imgs)
train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
val_imgs = [s for s in all_imgs if s['imageset'] == 'val']



num_anchors = len(C.anchor_box_scales['M1']) * len(C.anchor_box_ratios)
SSH = model.getModel()
optimizer = Adam(lr=1e-5)
SSH.summary()
SSH.load_weights('weights/imagenet_pretrained_model_ssh.vgg_2.hdf5', by_name=True)
# SSH = LoadWeights(SSH)
# print(SSH.get_weights())

SSH.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), \
										losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), \
										losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])


classes = {}

bbox_threshold = 0.8

visualise = True
cnt=0
for img_data in val_imgs:
	try:
		st = time.time()
		filepath = img_data['filepath']
		print (filepath)
		img = cv2.imread(filepath)

		X, ratio = format_img(img, C)

		if K.image_dim_ordering() == 'tf':
			X = np.transpose(X, (0, 2, 3, 1))

		[M1_Y1, M1_Y2, M2_Y1, M2_Y2, M3_Y1, M3_Y2] = SSH.predict(X)

		box_M1, prob_M1, module_M1 = ROI(M1_Y1, M1_Y2, C, K.image_dim_ordering(), 'M1')
		box_M2, prob_M2, module_M2 = ROI(M2_Y1, M2_Y2, C, K.image_dim_ordering(), 'M2')
		box_M3, prob_M3, module_M3 = ROI(M3_Y1, M3_Y2, C, K.image_dim_ordering(), 'M3')
		all_boxes = np.concatenate([box_M1, box_M2, box_M3])
		(all_boxes[:,0], all_boxes[:,1], all_boxes[:,2], all_boxes[:,3]) = get_real_coordinates(ratio, all_boxes[:,0], all_boxes[:,1], all_boxes[:,2], all_boxes[:,3])
		all_probs = np.concatenate([prob_M1, prob_M2, prob_M3])
		all_modules = np.concatenate([module_M1, module_M2, module_M3])
		R, probs, modules = roi_helpers.non_max_suppression_fast_module(all_boxes, all_probs, all_modules, overlap_thresh=0.3, max_boxes=300)
		f, r, p = f_measure_rpn(C, R, probs, img_data['bboxes'])
		for rpn in R:
			color = (0, 0, 255)
			(real_x1, real_y1, real_x2, real_y2) = (rpn[0], rpn[1], rpn[2], rpn[3])
			cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), color, 1)
		for gt in img_data['bboxes']:
			color = (0, 255, 0)
			(real_x1, real_y1, real_x2, real_y2) = (gt['x1'], gt['y1'], gt['x2'], gt['y2'])
			cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), color, 1)
		cv2.imwrite('test/out/{}'.format(str.split(img_data['filepath'], os.sep)[-1]),img)
		if(cnt>=1):
			break
		cnt+=1
	except Exception as e:
		print(e)
		continue


print(C.eval['rpn_rcount'], C.eval['rpn_pcount'], )
print('''
      Final RPN recall: {}, sum recall: {},
      Final RPN precision: {}, sum precision: {}
      '''.format(sum(C.eval['rpn_recall'])/len(C.eval['rpn_recall']), sum(C.eval['rpn_rcount'])/sum(C.eval['rpn_tcount']),
      		sum(C.eval['rpn_precision'])/len(C.eval['rpn_precision']), sum(C.eval['rpn_pcount'])/sum(C.eval['rpn_ocount'])))