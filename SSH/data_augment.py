import cv2
import numpy as np
import copy


def augment(img_data, C, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])

	# rows, cols = img.shape[:2]
	# M = np.float32([[1,0,C.right_horizontal_stride],[0,1,0]])
	# dst1 = cv2.warpAffine(img,M,(cols,rows))

	# M = np.float32([[1,0,C.left_horizontal_stride],[0,1,0]])
	# dst2 = cv2.warpAffine(img,M,(cols,rows))

	# M = np.float32([[1,0,0],[0,1,C.top_vertical_stride]])
	# dst3 = cv2.warpAffine(img,M,(cols,rows))

	# M = np.float32([[1,0,0],[0,1,C.bottom_vertical_stride]])
	# dst4 = cv2.warpAffine(img,M,(cols,rows))

	# return img, dst1, dst2, dst3, dst4
	return [img]
