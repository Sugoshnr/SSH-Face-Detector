from __future__ import division

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import random
import pprint
import sys
import time
import numpy as np
import keras
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
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, LambdaCallback
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, Flatten, Reshape
from keras.models import Model, load_model
import numpy as np
import math
import pickle_data_generators as data_generators

sys.setrecursionlimit(40000)

C = config.Config()

def LoadWeights(model):
	with open('mahyarnajibi_ssh_weights.pickle', 'rb') as f:
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
				w1 = w1[:,:,:,0:4:2]
				b = b[0:4:2]
			#w1 = w1[:,:,:,0:4::2]
			print (w1.shape, b.shape)
			model.get_layer(layer.name).set_weights([w1,b])
			print ("Copied weights for layer: ", layer.name)
		else:
			print ("Warning!!! No weights found for: ", layer.name)
	return model

def LoadVGGWeights(SSH, VGG):
	for l in VGG.layers:
		if 'input' in l.name or 'pool' in l.name:
			continue;
		if l.name in 'block3_conv2':
			return SSH
		weights = l.get_weights()
		block, layer = l.name.split('_')
		layer = 'conv' + block.replace('block','') + '_' + layer.replace('conv','')
		SSH.get_layer(layer).set_weights(weights)

def getLosses(clss, regr, img_data, C, module):
	R = roi_helpers.rpn_to_roi(clss, regr, C, K.image_dim_ordering(), module, use_regr=True, overlap_thresh=0.5, max_boxes=300)
	# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
	X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping, module)

	if X2 is None:
		return -1, -1	
	neg_samples = np.where(Y1[0, :, -1] == 1)
	pos_samples = np.where(Y1[0, :, -1] == 0)

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

num_anchors = len(C.anchor_box_scales['M1']) * len(C.anchor_box_ratios)

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

num_epochs = int(21000)

print('Starting training')

patience = 20
batchSize = 4

vis = True

def step_decay(epoch):
    if epoch > 250:
        epoch = 250

    initial_lrate = 0.00005
    drop = 0.1
    epochs_drop = 6
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    # lrate = initial_lrate * drop
    return lrate
lr_scheduler = LearningRateScheduler(step_decay)

VGG_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

SSH = model.getModel()
optimizer = SGD(lr=0.00005, momentum=0.9, decay=0.0005, nesterov=True)
SSH.summary()
# SSH = LoadWeights(SSH)
SSH.load_weights('imagenet_pretrained_2_model_ssh.vgg.hdf5')
# SSH = LoadVGGWeights(SSH, VGG_model)

SSH.compile(optimizer=optimizer,
			loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), \
										losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), \
										losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])

checkpoint = ModelCheckpoint('imagenet_pretrained_2_'+C.model_path, monitor = 'val_loss', save_best_only = True, verbose = 1)
earlystop = EarlyStopping(monitor = 'val_loss', patience = patience, verbose = 1)

def LogView(batch, logs = {}):
	print (logs.keys())

LogViewCallback = LambdaCallback(
    on_batch_end=LogView
    )

SSH.fit_generator(generator = data_gen_train,
				steps_per_epoch = len(train_imgs)//batchSize,
				validation_data = data_gen_val,
				validation_steps = len(val_imgs)//batchSize,
				epochs = num_epochs,
				max_queue_size = 20,
				callbacks = [checkpoint, earlystop, lr_scheduler],
				initial_epoch = 18)

