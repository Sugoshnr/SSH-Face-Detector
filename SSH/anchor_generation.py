import numpy as np
import cv2
import random
import copy
from PIL import Image, ImageDraw


def IoU(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, (xB - xA + 1)) * max(0, (yB - yA + 1))
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	if float(boxAArea + boxBArea - interArea) == 0
		return 0
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


def get_new_img_size(width, height, img_min_side=224):
	if width >= height:
		f = float(width) / img_min_side
		resized_height = int(height / f)
		resized_width = img_min_side
	else:
		f = float(height) / img_min_side
		resized_width = int(width / f)
		resized_height = img_min_side

	return resized_width, resized_height


def genAnchors(im, draw, gts):
	maxSize = 224
	strides = [8, 16, 32]
	predCnt = 0
	IoURatio = 0.5

	for stride in strides:
		x1, y1 = 0, 0; x2, y2 = stride-1, stride-1
		while(True):
			if(x2 > maxSize):
				x1 = 0
				x2 = stride-1
				y1 += stride-1
				y2 += stride-1
			if (y2 > maxSize):
				break
			# draw.rectangle([x1, y1, x2, y2])
			for gt in gts:
				if(IoU([x1, y1, x2, y2], gt) >= IoURatio):
					predCnt+=1
					print (IoU(gt, [x1, y1, x2, y2]))
					print ([x1, y1, x2, y2], gt)
					draw.rectangle([x1, y1, x2, y2], outline=(255,0,0))
					draw.rectangle(gt, outline=(0,255,0))
					print ('--------------------------')
			x1 += stride-1
			x2 += stride-1

		x1, y1 = (stride/4)-1, (stride/4)-1; x2, y2 = 3*(stride/4)-1, 3*(stride/4)-1
		while(True):
			if(x2 > maxSize):
				x1 = (stride/4)-1
				x2 = 3*(stride/4)-1
				y1 += stride-1
				y2 += stride-1
			if (y2 > maxSize):
				break
			# draw.rectangle([x1, y1, x2, y2])
			for gt in gts:
				if(IoU([x1, y1, x2, y2], gt) >= IoURatio):
					predCnt+=1
					print (IoU(gt, [x1, y1, x2, y2]))
					print ([x1, y1, x2, y2], gt)
					draw.rectangle([x1, y1, x2, y2], outline=(255,0,0))
					draw.rectangle(gt, outline=(0,255,0))
					print ('--------------------------')
			x1 += stride-1
			x2 += stride-1
	print (predCnt)
	im.show()



def getAnchors(original_image, gt):
	height, width = original_image.shape[:2]
	(resized_width, resized_height) = get_new_img_size(width, height)
	print(type(original_image))
	final_image = np.zeros((224,224,3))
	original_image = cv2.resize(original_image,(resized_width, resized_height), interpolation = cv2.INTER_CUBIC)
	# img = np.asarray(original_image)
	# Image.fromarray(img).show()
	print(original_image.shape[:2])
	final_image[:resized_height, :resized_width, :] = original_image
	print(original_image)
	cv2.imshow('img1', original_image)
	print (final_image)
	cv2.imshow('img', final_image.astype('float32'))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# im = Image.fromarray(final_image.astype('uint8'))
	# draw = ImageDraw.Draw(im)
	# for x in gt:
	# 	x[0] *= (resized_width / float(width))
	# 	x[2] *= (resized_width / float(width))
	# 	x[1] *= (resized_height / float(height))
	# 	x[3] *= (resized_height / float(height))
		# draw.rectangle(x, outline=(0, 0, 255))
	# genAnchors(im, draw, gt)
	# print (IoU(rect, gt[0]))
	# im.show()

original_image = cv2.imread("image.jpg")
f = open('gt.txt', 'r')
gt = []
for line in f:
	x = line.split()
	bb = [float(x[0]), float(x[1]), float(x[0]) + float(x[2]), float(x[1]) + float(x[3])]
	gt.append(bb)
getAnchors(original_image, gt)


