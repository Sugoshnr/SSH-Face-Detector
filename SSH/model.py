import keras
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, Flatten, Reshape
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
import numpy as np

input_shape = (224, 224, 3)
inputs = Input(shape=input_shape)

# img = load_img('WIDER/WIDER/WIDER_train/images/0--Parade/0_Parade_marchingband_1_6.jpg')
# img.show()
# print(img.size, img.mode)
# x = img_to_array(img)
# print (x.shape)
# x = x.reshape((1,) + x.shape)
# print (x.shape)

	
def VGG16_base():
	model = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
	model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
	model = MaxPooling2D((2,2), strides=(2,2))(model)

	model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)
	model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)
	model = MaxPooling2D((2,2), strides=(2,2))(model)

	model = Conv2D(256, (3, 3), activation='relu', padding='same')(model)
	model = Conv2D(256, (3, 3), activation='relu', padding='same')(model)
	model = Conv2D(256, (3, 3), activation='relu', padding='same')(model)
	model = MaxPooling2D((2,2), strides=(2,2))(model)

	model = Conv2D(512, (3, 3), activation='relu', padding='same')(model)
	model = Conv2D(512, (3, 3), activation='relu', padding='same')(model)
	model = Conv2D(512, (3, 3), activation='relu', padding='same')(model)
	
	return model

def ContextModel(model, stride, X):
	# Removed strides
	context_conv_1 = Conv2D(int(X/2), (3,3), activation='relu', padding='same')(model)
	context_conv_2 = Conv2D(int(X/2), (3,3), activation='relu', padding='same')(context_conv_1)
	context_conv_3 = Conv2D(int(X/2), (3,3), activation='relu', padding='same')(context_conv_1)
	context_conv_3 = Conv2D(int(X/2), (3,3), activation='relu', padding='same')(context_conv_3)
	context_model = keras.layers.concatenate([context_conv_2, context_conv_3])
	return context_model

def DetectionModel(model, name):
	if name == 'M1':
		stride = (8, 8)
		X = 128
	if name == 'M2':
		stride = (16, 16)
		X = 256
	if name == 'M3':
		stride = (32, 32)
		X = 256
	detection_conv_model = Conv2D(X, (3, 3), activation='relu', padding='same')(model)
	detection_context_model = ContextModel(model, stride, X)
	detection_model = keras.layers.concatenate([detection_conv_model, detection_context_model])
	classScores = Conv2D(4, (1,1), padding='same')(detection_model)
	regressorScores = Conv2D(8, (1,1), padding='same')(detection_model)
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
	model = MaxPooling2D((2,2), strides=(2,2))(model)
	model = Conv2D(512, (3, 3), activation='relu', padding='same')(model)
	model = Conv2D(512, (3, 3), activation='relu', padding='same')(model)
	model = Conv2D(512, (3, 3), activation='relu', padding='same')(model)
	return model


baseModel = VGG16_base()

VGGModel = auxVGG(baseModel)

# M2 Detection Module
DetectionModulelM2 = DetectionModel(VGGModel, 'M2')



# M3 Detection Module
VGGModel_pool = MaxPooling2D((2,2), strides=(2,2))(VGGModel)
DetectionModulelM3 = DetectionModel(VGGModel_pool, 'M3')



# M1 Detection Module
M1_dimReduction_1 = Conv2D(128, (1, 1), activation='relu', padding='same')(baseModel)
M1_dimReduction_2 = Conv2D(128, (1, 1), activation='relu', padding='same')(VGGModel)
M1_dimReduction_2 = keras.layers.UpSampling2D(size=(2, 2))(M1_dimReduction_2)
M1_elementWiseSum = keras.layers.Add()([M1_dimReduction_1, M1_dimReduction_2])
M1_conv = Conv2D(128, (3, 3), activation='relu', padding='same')(M1_elementWiseSum)

DetectionModulelM1 = DetectionModel(M1_conv, 'M1')

model = Model(inputs=inputs, outputs=list((np.asarray([DetectionModulelM1, DetectionModulelM2, DetectionModulelM3]).flatten())))

model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.summary()