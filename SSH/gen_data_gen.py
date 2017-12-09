import config
from simple_parser_gen import get_data
# import data_generators_gen
import numpy as np
from keras import backend as K
import os
import cPickle
import cv2
import data_generators_gen_2 as data_generators_gen

# mine
def get_new_img_size(width, height, img_min_side=1024):
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


C = config.Config()

all_imgs, classes_count, class_mapping = get_data('')
train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
val_imgs = [s for s in all_imgs if s['imageset'] == 'val']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

data_gen_train = data_generators_gen.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators_gen.get_anchor_gt(val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

# Pickle Dump 1
cnt = 0
while cnt < len(train_imgs):
	X, Y, img_data = next(data_gen_train)
	p = img_data['filepath'].split(os.sep)[:-1]
	f = img_data['filepath'].split(os.sep)[-1]+'.bo5pickle'
	final_path = (os.path.join(os.sep.join(p), f))
	# print(p, f, final_path)
	# print(img_data.keys())
	# print((img_data[img_data['stride']]))
	# print(getsizeof(img_data))
	# # A = np.asarray(img_data)
	# # np_compression.compress(A)
	cPickle.dump(img_data, open(final_path, "wb" ))
	cnt+=1
	print(cnt)
	
cnt = 0
while cnt < len(val_imgs):
	X, Y, img_data = next(data_gen_val)
	p = img_data['filepath'].split(os.sep)[:-1]
	f = img_data['filepath'].split(os.sep)[-1]+'.bo5pickle'
	final_path = (os.path.join(os.sep.join(p), f))
	# print(p, f, final_path)
	# print(img_data.keys())
	# print((img_data[img_data['stride']]))
	# print(getsizeof(img_data))
	# # A = np.asarray(img_data)
	# # np_compression.compress(A)
	cPickle.dump(img_data, open(final_path, "wb" ))
	cnt+=1
	print(cnt)




# Pickle Dump 2
# cnt = 0
# while cnt < len(train_imgs):
# 	X, Y, img_data = next(data_gen_train)
# 	p = img_data['filepath'].split(os.sep)[:-1]
# 	f = img_data['filepath'].split(os.sep)[-1]+'.pickle'
# 	final_path = (os.path.join(os.sep.join(p), f))
# 	print(p,f, final_path)
# 	pickle.dump(C.representation, open( final_path, "wb" ))
# 	cnt+=1
# 	print(cnt)
	
# cnt = 0
# while cnt < len(val_imgs):
# 	X, Y, img_data = next(data_gen_val)
# 	p = img_data['filepath'].split(os.sep)[:-1]
# 	f = img_data['filepath'].split(os.sep)[-1]+'.pickle'
# 	final_path = (os.path.join(os.sep.join(p), f))
# 	print(p,f, final_path)
# 	pickle.dump(C.representation, open( final_path, "wb" ))
# 	cnt+=1
# 	print(cnt)


# Resize and write
# cnt = 0
# for img_data in train_imgs:
# 	# X, Y, img_data = next(data_gen_train)
# 	img_data_aug = img_data
# 	x_img = cv2.imread(img_data_aug['filepath'])
# 	(width, height) = (img_data_aug['width'], img_data_aug['height'])
# 	(rows, cols, _) = x_img.shape
# 	print(cnt)
# 	# print(img_data_aug['filepath'])
# 	# print(width, height)
# 	# print(cols, rows)
# 	assert cols == width
# 	assert rows == height

# 	# get image dimensions for resizing
# 	(resized_width, resized_height) = get_new_img_size(width, height, C.im_size)
# 	# resize the image so that smalles side is length = 600px
# 	x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
# 	cv2.imwrite(img_data_aug['filepath'], x_img)
# 	cnt+=1
# 	# print('-----------------------------')

# cnt = 0
# for img_data in val_imgs:
# 	# X, Y, img_data = next(data_gen_train)
# 	img_data_aug = img_data
# 	x_img = cv2.imread(img_data_aug['filepath'])
# 	(width, height) = (img_data_aug['width'], img_data_aug['height'])
# 	(rows, cols, _) = x_img.shape
# 	print(cnt)
# 	# print(img_data_aug['filepath'])
# 	# print(width, height)
# 	# print(cols, rows)
# 	assert cols == width
# 	assert rows == height

# 	# get image dimensions for resizing
# 	(resized_width, resized_height) = get_new_img_size(width, height, C.im_size)
# 	# resize the image so that smalles side is length = 600px
# 	x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
# 	cv2.imwrite(img_data_aug['filepath'], x_img)
# 	cnt+=1



# Copy original back
# cnt = 0
# for img_data in train_imgs:
# 	# X, Y, img_data = next(data_gen_train)
# 	filepath = str(img_data['filepath'])
# 	new_filepath_split = str.split(filepath, os.sep)
# 	new_filepath = os.path.join(new_filepath_split[0], new_filepath_split[1]+'_copy', os.sep.join(new_filepath_split[1:]))
# 	os.system("cp "+new_filepath+" "+filepath)
# 	cnt+=1
# 	print(cnt)
# 	# print('-----------------------------')
# cnt = 0
# print('Begin')
# for img_data in val_imgs:
# 	# X, Y, img_data = next(data_gen_train)
# 	filepath = str(img_data['filepath'])
# 	new_filepath_split = str.split(filepath, os.sep)
# 	new_filepath = os.path.join(new_filepath_split[0], new_filepath_split[1]+'_copy', os.sep.join(new_filepath_split[1:]))
# 	os.system("cp "+new_filepath+" "+filepath)
# 	# cnt+=1
# 	# print(cnt)
# 	# print('-----------------------------')
# print('End')
	
# pickle.dump(C.representation, open( "WIDER_target_anchors.pickle", "wb" ))
