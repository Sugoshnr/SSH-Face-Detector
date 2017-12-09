from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
import data_augment
import threading
import itertools
import traceback

def get_img_output_length(width, height, rpn_stride):
    def get_output_length(input_length):
        return input_length//rpn_stride

    return get_output_length(width), get_output_length(height)

def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


# frcnn
# def iou(a, b):
# 	# a and b should be (x1,y1,x2,y2)

# 	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
# 		return 0.0

# 	area_i = intersection(a, b)
# 	area_u = union(a, b, area_i)

# 	return float(area_i) / float(area_u + 1e-6)

# mine
def IoU(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, (xB - xA + 1)) * max(0, (yB - yA + 1))
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	if float(boxAArea + boxBArea - interArea) == 0:
		return 0
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


# frcnn
# def get_new_img_size(width, height, img_min_side=600):
# 	if width <= height:
# 		f = float(img_min_side) / width
# 		resized_height = int(f * height)
# 		resized_width = img_min_side
# 	else:
# 		f = float(img_min_side) / height
# 		resized_width = int(f * width)
		# resized_height = img_min_side

# 	return resized_width, resized_height

# mine
def get_new_img_size(width, height, img_min_side=1344):
	if width >= height:
		f = float(width) / img_min_side
		resized_height = int(height / f)
		resized_width = img_min_side
		if not int(resized_height) % 2 == 0:
			resized_height+=1
	else:
		f = float(height) / img_min_side
		resized_width = int(width / f)
		resized_height = img_min_side
		if not int(resized_width) % 2 == 0:
			resized_height+=1

	return resized_width, resized_height


class SampleSelector:
	def __init__(self, class_count):
		# ignore classes that have zero samples
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		self.class_cycle = itertools.cycle(self.classes)
		self.curr_class = next(self.class_cycle)

	def skip_sample_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True

def findBest(C, module, best, resized_width, resized_height, img_data, img, stride):
	best_anchor_for_bbox = best['anchor']
	best_iou_for_bbox = best['iou']
	best_x_for_bbox = best['x']
	best_dx_for_bbox = best['dx']
	num_anchors_for_bbox = best['num_anchors']
	anchor_sizes = C.anchor_box_scales[module]
	anchor_ratios = C.anchor_box_ratios
	num_anchors = len(anchor_sizes) * len(anchor_ratios)	
	n_anchratios = len(anchor_ratios)

	(output_width, output_height) = get_img_output_length(resized_width, resized_height, C.rpn_stride[module])
	
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
	y_is_box_valid = np.ones((output_height, output_width, num_anchors))
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))
	# img = cv2.imread(img_data['filepath'])
	for idx in range(num_anchors_for_bbox.shape[0]):
		# if num_anchors_for_bbox[idx] == 0:
		if best['module'][idx] == int(module[1]):
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1:
				continue
			# print((best_x_for_bbox[idx,0], best_x_for_bbox[idx,2]), (best_x_for_bbox[idx,1], best_x_for_bbox[idx,3]))
			# print('iou = ', best_iou_for_bbox[idx])
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]
	# print (cnt)
	for anchors in best['neutral_anchors'][module]:
		y_is_box_valid[anchors[0], anchors[1], anchors[2]] = 0
	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)
	# f = img_data['filepath'].split(os.sep)[-1]
	# print(f)
	img_data[stride][module] = [np.copy(y_is_box_valid), np.copy(y_rpn_overlap), np.copy(y_rpn_regr)]
	# y_is_box_valid, y_rpn_overlap, y_rpn_regr = C.representation[f][module]
	# C.representation[f][module] = [y_is_box_valid, y_rpn_overlap, y_rpn_regr]

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
	neutral_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 0))

	# print('{} {} pos, neg, neut = {} {} {}'.format(module, stride, len(pos_locs[0]), len(neg_locs[0]), len(neutral_locs[0])))

	num_pos = len(pos_locs[0])
	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions. We also limit it to 256 regions.

	num_regions = 256

	if len(pos_locs[0]) > num_regions/2:
		val_locs = random.sample(range(len(pos_locs[0])), int(len(pos_locs[0]) - int(num_regions/2)))
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), int(len(neg_locs[0]) - int(num_pos)))
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0
	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))
	neutral_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 0))

	# print('pos, neg, neut = ', len(pos_locs[0]), len(neg_locs[0]), len(neutral_locs[0]))

	return y_rpn_cls, y_rpn_regr


def calc_rpn(C, img_data, width, height, resized_width, resized_height, module, img, best, stride):
# def calc_rpn(C, img_data, width, height, resized_width, resized_height, module, img):

	# print(module)
	downscale = float(C.rpn_stride[module])
	# anchor_sizes = C.anchor_box_scales
	anchor_sizes = C.anchor_box_scales[module]
	anchor_ratios = C.anchor_box_ratios
	num_anchors = len(anchor_sizes) * len(anchor_ratios)	
	# calculate the output map size based on the network architecture
	(output_width, output_height) = get_img_output_length(resized_width, resized_height, C.rpn_stride[module])
	# (output_width, output_height) = C.img_output_length[module]
	# print(anchor_sizes, (output_width, output_height), num_anchors)
	n_anchratios = len(anchor_ratios)

	# initialise empty output objectives
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

	y_rpn_cls = [0]

	num_bboxes = len(img_data['bboxes'])
	
	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
	
	best_anchor_for_bbox = best['anchor']
	best_iou_for_bbox = best['iou']
	best_x_for_bbox = best['x']
	best_dx_for_bbox = best['dx']
	num_anchors_for_bbox = best['num_anchors']
	# print(best['anchor'].shape, best['iou'].shape, best['x'].shape, best['dx'].shape)
	
	# get the GT box coordinates, and resize to account for image resizing
	gta = np.zeros((num_bboxes, 4))
	
	for bbox_num, bbox in enumerate(img_data['bboxes']):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
		gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
		gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
		gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
		if((gta[bbox_num, 1] - gta[bbox_num, 0]) <= 0 or (gta[bbox_num, 3] - gta[bbox_num, 2]) <= 0):
			raise Exception('width or height <=0')
	if stride == 'right':
		gta[:, 0]+=C.right_horizontal_stride
		gta[:, 1]+=C.right_horizontal_stride
	elif stride == 'left':
		gta[:, 0]+=C.left_horizontal_stride
		gta[:, 1]+=C.left_horizontal_stride
	elif stride == 'top':
		gta[:, 2]+=C.top_vertical_stride
		gta[:, 3]+=C.top_vertical_stride
	elif stride == 'bottom':
		gta[:, 2]+=C.bottom_vertical_stride
		gta[:, 3]+=C.bottom_vertical_stride
	for bbox_num, bbox in enumerate(img_data['bboxes']):
		if True:
			x1_gt, x2_gt, y1_gt, y2_gt = map(int, gta[bbox_num, :])
			# print((x1_gt, y1_gt), (x2_gt, y2_gt))
			cv2.rectangle(img, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 255, 0), 1)
	

	# resized_width = resized_height = C.im_size
	# rpn ground truth
	for anchor_size_idx in range(len(anchor_sizes)):
		# for anchor_ratio_idx in range(n_anchratios):
		anchor_ratio_idx = 0
		anchor_x = anchor_sizes[anchor_size_idx] * 1
		anchor_y = anchor_sizes[anchor_size_idx] * 1	
		
		for ix in range(output_width):					
			# x-coordinates of the current anchor box	
			x1_anc = downscale * (ix + 0.5) - anchor_x / 2
			x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
			
			# ignore boxes that go across image boundaries					
			if x1_anc < 0 or x2_anc > resized_width:
				continue
				
			for jy in range(output_height):

				# y-coordinates of the current anchor box
				y1_anc = downscale * (jy + 0.5) - anchor_y / 2
				y2_anc = downscale * (jy + 0.5) + anchor_y / 2
				# ignore boxes that go across image boundaries
				if y1_anc < 0 or y2_anc > resized_height:
					continue

				# bbox_type indicates whether an anchor should be a target 
				bbox_type = 'neg'

				# this is the best IOU for the (x,y) coord and the current anchor
				# note that this is different from the best IOU for a GT bbox
				best_iou_for_loc = 0.0
				# cv2.rectangle(img, (int(x1_anc), int(y1_anc)), (int(x2_anc), int(y2_anc)), (0, 0, 255))
				for bbox_num in range(num_bboxes):
					# get IOU of the current GT box and the current anchor box
					curr_iou = IoU([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
					# calculate the regression targets if they will be needed
					if curr_iou > best_iou_for_bbox[bbox_num]:# or curr_iou > C.rpn_max_overlap:
						cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
						cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
						cxa = (x1_anc + x2_anc)/2.0
						cya = (y1_anc + y2_anc)/2.0

						tx = (cx - cxa) / (x2_anc - x1_anc)
						ty = (cy - cya) / (y2_anc - y1_anc)
						tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
						th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
						
						best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
						best_iou_for_bbox[bbox_num] = curr_iou
						best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
						best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]
						best['module'][bbox_num] = int(module[1])

						bbox_type = 'pos'
						best['type'][bbox_num] = 1
						num_anchors_for_bbox[bbox_num] += 1
						# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
						if curr_iou > best_iou_for_loc:
							best_iou_for_loc = curr_iou
							best_regr = (tx, ty, tw, th)


					# if img_data['bboxes'][bbox_num]['class'] != 'bg':

					# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
					# if curr_iou > best_iou_for_bbox[bbox_num]:
						
					# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
					# if curr_iou > C.rpn_max_overlap:
						
					# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
					if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
						# gray zone between neg and pos
						if bbox_type != 'pos':
							bbox_type = 'neutral'
				if bbox_type == 'neutral':
					best['neutral_anchors'][module].append([jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx])
				# turn on or off outputs depending on IOUs
				# if bbox_type == 'neg':
				# 	# print('Negative')
				# 	y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
				# 	y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
				# elif bbox_type == 'neutral':
				# 	# print('neutral')
				# 	y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
				# 	y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
				# elif bbox_type == 'pos':
				# 	cv2.rectangle(img, (int(x1_anc), int(y1_anc)), (int(x2_anc), int(y2_anc)), (255, 0, 0), thickness=1)
				# 	y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
				# 	y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
				# 	start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
				# 	y_rpn_regr[jy, ix, start:start+4] = best_regr


				# if bbox_type == 'pos':
					# cv2.rectangle(img, (int(x1_anc), int(y1_anc)), (int(x2_anc), int(y2_anc)), (255, 0, 0), thickness=1)
	# if C.diagnose and module == 'M3':
	# 	for bbox_num in range(num_bboxes):
	# 		color = {'pos': (255, 255, 255), 'neutral': (255, 255, 255), 'neg': (0, 0, 255)}
	# 		x1_ab, x2_ab, y1_ab, y2_ab = map(int, best_x_for_bbox[bbox_num])
	# 		cv2.rectangle(img, (int(x1_ab), int(y1_ab)), (int(x2_ab), int(y2_ab)), color['pos'], thickness=1)
	# 		if best['type'][bbox_num] == 1:
	# 			cv2.rectangle(img, (int(x1_ab), int(y1_ab)), (int(x2_ab), int(y2_ab)), color['neg'], thickness=1)
				

	# if C.diagnose:
	# 	import os
	# 	p = img_data['filepath'].split(os.sep)[:-2]
	# 	f = img_data['filepath'].split(os.sep)[-1]
	# 	# f = (img_data['filepath'].split(os.sep)[-1])[:-5]+'_600'+(img_data['filepath'].split(os.sep)[-1])[-4:]
	# 	# print((os.path.join(os.sep.join(p), 'diagnose2', f)))
	# 	height, width = img.shape[:2]
	# 	cv2.imwrite(os.path.join(os.sep.join(p), 'diagnose2', f), img)
		
	best['anchor'] = best_anchor_for_bbox
	best['iou'] = best_iou_for_bbox
	best['x'] = best_x_for_bbox
	best['dx'] = best_dx_for_bbox
	best['num_anchors'] = num_anchors_for_bbox
	# we ensure that every bbox has at least one positive RPN region
	if module == 'M3':
		# C.representation[img_data['filepath']] = {}
		img_data[stride] = {}
		y_rpn_cls_M1, y_rpn_regr_M1 = findBest(C, 'M1', best, resized_width, resized_height, img_data, img, stride)
		y_rpn_cls_M2, y_rpn_regr_M2 = findBest(C, 'M2', best, resized_width, resized_height, img_data, img, stride)
		y_rpn_cls_M3, y_rpn_regr_M3 = findBest(C, 'M3', best, resized_width, resized_height, img_data, img, stride)
		# print(len(np.where(best['module']>0)[0]))
		# print(stride, num_bboxes)
		# cv2.imwrite(stride+'1.jpg',img)
	
		return np.copy(y_rpn_cls_M1), np.copy(y_rpn_regr_M1), np.copy(y_rpn_cls_M2), np.copy(y_rpn_regr_M2), \
				np.copy(y_rpn_cls_M3), np.copy(y_rpn_regr_M3), len(np.where(best['module']>0)[0])

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr), img, best


class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)		

	
def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g

def get_anchor_gt(all_img_data, class_count, C, backend, mode='train'):

	# The following line is not useful with Python 3.5, it is kept for the legacy
	# all_img_data = sorted(all_img_data)
	
	sample_selector = SampleSelector(class_count)

	while True:
		if mode == 'train':
			np.random.shuffle(all_img_data)

		for img_data in all_img_data:
			try:

				if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
					continue
				augmented_data = {}
				max_count = -1
				best_stride = 'original'
				# read in image, and optionally add augmentation
				images = data_augment.augment(img_data, C)
				for image_idx in range(len(images)):
					# Augment bboxes
					img_data_aug = copy.deepcopy(img_data)
					x_img = copy.deepcopy(images[image_idx])
					stride = 'original'
					if(image_idx == 1):
						stride = 'right'
					elif(image_idx == 2):
						stride = 'left'
					elif(image_idx == 3):
						stride = 'top'
					elif(image_idx == 4):
						stride = 'bottom'
					img_data_aug['stride'] = stride
					# x_img = cv2.imread(img_data_aug['filepath'])
					(width, height) = (img_data_aug['width'], img_data_aug['height'])
					(rows, cols, _) = x_img.shape

					assert cols == width
					assert rows == height

					# get image dimensions for resizing
					(resized_width, resized_height) = get_new_img_size(width, height, C.im_size)
					num_bboxes = len(img_data_aug['bboxes'])
					best = {}
					best['anchor'] = -1*np.ones((num_bboxes, 4)).astype(int)
					best['iou'] = C.rpn_max_overlap * np.ones(num_bboxes).astype(np.float32)
					best['x'] = np.zeros((num_bboxes, 4)).astype(int)
					best['dx'] = np.zeros((num_bboxes, 4)).astype(np.float32)
					best['num_anchors'] = np.zeros(num_bboxes).astype(int)
					best['module'] = np.zeros(num_bboxes).astype(int)
					best['type'] = np.zeros(num_bboxes).astype(int)
					best['neutral_anchors'] = {'M1': [], 'M2': [], 'M3': []}
					# resize the image so that smalles side is length = 600px
					x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
					# final_image = np.zeros((C.im_size, C.im_size, 3))
					# final_image[:resized_height, :resized_width, :] = x_img
					# x_img = final_image
					# TODO Remove hardcode
					# print('shape',x_img.shape)
					try:
						y_rpn_cls, y_rpn_regr, x_img, best = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, 'M1', np.copy(x_img), best, stride)
						y_rpn_cls2, y_rpn_regr2, x_img, best = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, 'M2', np.copy(x_img), best, stride)
						y_rpn_cls_M1, y_rpn_regr_M1, y_rpn_cls_M2, y_rpn_regr_M2, y_rpn_cls_M3, y_rpn_regr_M3, anchor_count = calc_rpn( \
							C, img_data_aug, width, height, resized_width, resized_height, 'M3', np.copy(x_img), best, stride \
						)
						# y_rpn_cls_M1, y_rpn_regr_M1 = findBest(C, 'M1', best, resized_width, resized_height, img_data)
						# y_rpn_cls_M2, y_rpn_regr_M2 = findBest(C, 'M2', best, resized_width, resized_height, img_data)
						# y_rpn_cls_M3, y_rpn_regr_M3 = findBest(C, 'M3', best, resized_width, resized_height, img_data)
			
					except Exception as e:
						print('Failure',e)
						print(traceback.format_exc())
						continue
					
					# Zero-center by mean pixel, and preprocess image
			
					x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
					x_img = x_img.astype(np.float32)
					x_img[:, :, 0] -= C.img_channel_mean[0]
					x_img[:, :, 1] -= C.img_channel_mean[1]
					x_img[:, :, 2] -= C.img_channel_mean[2]
					x_img /= C.img_scaling_factor

					x_img = np.transpose(x_img, (2, 0, 1))
					x_img = np.expand_dims(x_img, axis=0)

					y_rpn_regr_M1[:, y_rpn_regr_M1.shape[1]//2:, :, :] *= C.std_scaling
					y_rpn_regr_M2[:, y_rpn_regr_M2.shape[1]//2:, :, :] *= C.std_scaling
					y_rpn_regr_M3[:, y_rpn_regr_M3.shape[1]//2:, :, :] *= C.std_scaling

					if backend == 'tf':
						x_img = np.transpose(x_img, (0, 2, 3, 1))
						y_rpn_cls_M1 = np.transpose(y_rpn_cls_M1, (0, 2, 3, 1))
						y_rpn_regr_M1 = np.transpose(y_rpn_regr_M1, (0, 2, 3, 1))
						y_rpn_cls_M2 = np.transpose(y_rpn_cls_M2, (0, 2, 3, 1))
						y_rpn_regr_M2 = np.transpose(y_rpn_regr_M2, (0, 2, 3, 1))
						y_rpn_cls_M3 = np.transpose(y_rpn_cls_M3, (0, 2, 3, 1))
						y_rpn_regr_M3 = np.transpose(y_rpn_regr_M3, (0, 2, 3, 1))
					
					augmented_data[stride] = [np.copy(x_img), \
											[np.copy(y_rpn_cls_M1), np.copy(y_rpn_regr_M1), \
											np.copy(y_rpn_cls_M2), np.copy(y_rpn_regr_M2), \
											np.copy(y_rpn_cls_M3), np.copy(y_rpn_regr_M3)], \
											img_data_aug]
					if(anchor_count>max_count):
						max_count = anchor_count
						best_stride = stride
				# yield np.copy(x_img), \
				# [np.copy(y_rpn_cls_M1), np.copy(y_rpn_regr_M1), \
				# np.copy(y_rpn_cls_M2), np.copy(y_rpn_regr_M2), \
				# np.copy(y_rpn_cls_M3), np.copy(y_rpn_regr_M3)], \
				# img_data_aug
				yield augmented_data[best_stride]
			except Exception as e:
				print(e)
				print(traceback.format_exc())
				continue