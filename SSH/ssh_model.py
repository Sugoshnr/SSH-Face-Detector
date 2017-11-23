import keras
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, Flatten, Reshape, Conv2DTranspose, Cropping2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
import numpy as np
# import cPickle
import config

# with open('ssh_weights.pickle', 'rb') as f:
# 	weights = cPickle.load(f)
# 	new_weights = dict()
# 	for name, w in weights.iteritems():
# 		if "@" in name:
# 			name = name.replace("@", "_")
# 		new_weights[name] = w
# 	weights = new_weights


C = config.Config()

input_shape = (C.im_size, C.im_size, 3)
inputs = Input(shape=input_shape)

# img = load_img('WIDER/WIDER/WIDER_train/images/0--Parade/0_Parade_marchingband_1_6.jpg')
# img.show()
# print(img.size, img.mode)
# x = img_to_array(img)
# print (x.shape)
# x = x.reshape((1,) + x.shape)
# print (x.shape)

	
def VGG16_base():
	model = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
	model = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(model)
	model = MaxPooling2D((2,2), strides=(2,2), name='pool1')(model)

	model = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(model)
	model = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(model)
	model = MaxPooling2D((2,2), strides=(2,2), name='pool2')(model)

	model = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(model)
	model = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(model)
	model = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(model)
	model = MaxPooling2D((2,2), strides=(2,2), name='pool3')(model)

	model = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(model)
	model = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(model)
	model = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(model)
	
	return model

def ContextModel(model, X, context_conv_name):
	# Removed strides
	context_conv_1 = Conv2D(int(X/2), (3,3), activation='relu', padding='same', name=context_conv_name[0])(model)
	context_conv_2 = Conv2D(int(X/2), (3,3), activation='relu', padding='same', name=context_conv_name[1])(context_conv_1)
	context_conv_3 = Conv2D(int(X/2), (3,3), activation='relu', padding='same', name=context_conv_name[2])(context_conv_1)
	context_conv_3 = Conv2D(int(X/2), (3,3), activation='relu', padding='same', name=context_conv_name[3])(context_conv_3)
	context_model = keras.layers.concatenate([context_conv_2, context_conv_3], name=context_conv_name[4])
	return context_model

def DetectionModel(model, name, detection_conv_name, context_conv_name):
	if name == 'M1':
		# stride = (8, 8)
		X = 128
	if name == 'M2':
		# stride = (16, 16)
		X = 256
	if name == 'M3':
		# stride = (32, 32)
		X = 256
	detection_conv_model = Conv2D(X, (3, 3), activation='relu', padding='same', name=detection_conv_name[0])(model)
	detection_context_model = ContextModel(model, X, context_conv_name)
	detection_model = keras.layers.concatenate([detection_conv_model, detection_context_model], name=detection_conv_name[1]+'_concat')
	classScores = Conv2D(2, (1,1), padding='same', activation='softmax', name=detection_conv_name[2])(detection_model)
	regressorScores = Conv2D(8, (1,1), padding='same', activation='linear', name=detection_conv_name[3])(detection_model)
	# shape_new_1 = (int((classScores_conv.shape.__getitem__(1)*classScores_conv.shape.__getitem__(2)*classScores_conv.shape.__getitem__(3))),)
	# shape_new_2 = (int((regressorScores_conv.shape.__getitem__(1)*regressorScores_conv.shape.__getitem__(2)*regressorScores_conv.shape.__getitem__(3))),)
	# classScores =keras.layers.Reshape(shape_new_1)(classScores_conv)
	# regressorScores = keras.layers.Reshape(shape_new_2, name='regressor_output_'+name)(regressorScores_conv)
	# print (name+' class Scores conv '+str(classScores_conv))
	# print (name+' class Scores '+str(classScores))
	# print (name+' regress Scores conv '+str(regressorScores_conv))
	# print (name+' regress Scores '+str(regressorScores))
	return [classScores, regressorScores]

def auxVGG(model):
	model = MaxPooling2D((2,2), strides=(2,2), name='pool4')(model)
	model = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(model)
	model = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(model)
	model = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(model)
	return model

def getModel():
	baseModel = VGG16_base()

	VGGModel = auxVGG(baseModel)

	# M2 Detection Module
	detection_conv_name = ['m2_ssh_3x3', 'm2_ssh_output', 'm2_ssh_cls_score', 'm2_ssh_bbox_pred']
	context_conv_name = ['m2_ssh_dimred', 'm2_ssh_5x5', 'm2_ssh_7x7-1', 'm2_ssh_7x7', 'm2_ssh_output']
	DetectionModulelM2 = DetectionModel(VGGModel, 'M2', detection_conv_name, context_conv_name)



	# M3 Detection Module
	detection_conv_name = ['m3_ssh_3x3', 'm3_ssh_output', 'm3_ssh_cls_score', 'm3_ssh_bbox_pred']
	context_conv_name = ['m3_ssh_dimred', 'm3_ssh_5x5', 'm3_ssh_7x7-1', 'm3_ssh_7x7', 'm3_ssh_output']
	VGGModel_pool = MaxPooling2D((2,2), strides=(2,2), name='pool6')(VGGModel)
	DetectionModulelM3 = DetectionModel(VGGModel_pool, 'M3', detection_conv_name, context_conv_name)

	# M1 Detection Module
	detection_conv_name = ['m1_ssh_3x3', 'm1_ssh_output', 'm1_ssh_cls_score', 'm1_ssh_bbox_pred']
	context_conv_name = ['m1_ssh_dimred', 'm1_ssh_5x5', 'm1_ssh_7x7-1', 'm1_ssh_7x7', 'm1_ssh_output']

	M1_dimReduction_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv4_128')(baseModel)
	M1_dimReduction_2 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv5_128')(VGGModel)
	M1_dimReduction_2 = keras.layers.UpSampling2D(size=(2, 2), name='conv5_128_upsam')(M1_dimReduction_2)
	M1_dimReduction_2 = Conv2DTranspose(128, kernel_size=(4, 4), name='conv5_128_up', padding='same', strides=(2,2))(M1_dimReduction_2)
	## TODO: Experiment on this cropping 2d offsets
	M1_dimReduction_2 = Cropping2D(cropping=((0, int(C.im_size/8)), (0, int(C.im_size/8))))(M1_dimReduction_2)
	M1_elementWiseSum = keras.layers.Add(name='conv4_fuse')([M1_dimReduction_1, M1_dimReduction_2])
	M1_conv = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4_fuse_final')(M1_elementWiseSum)

	DetectionModulelM1 = DetectionModel(M1_conv, 'M1', detection_conv_name, context_conv_name)

	model = Model(inputs=inputs, outputs=list((np.asarray([DetectionModulelM1, DetectionModulelM2, DetectionModulelM3]).flatten())))
	return model

	# for layer in model.layers:
	# 	if weights.has_key(layer.name):
	# 		w, b = weights[layer.name]
	# 		print w.shape, b.shape
	# ## TODO: Check transpose axes orderings, specifically 3,2,1,0
	# 		w1 = np.transpose(w, (2,3,1,0))
	# 		model.get_layer(layer.name).set_weights([w1,b])
	# 		print "Copied weights for layer: ", layer.name
	# 	else:
	# 		print "Warning!!! No weights found for: ", layer.name

	# model.compile(optimizer='sgd', loss='categorical_crossentropy')
	# model.summary()
