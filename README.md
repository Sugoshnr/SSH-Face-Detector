# SSH-Face-Detector
SSH Face Detector in Keras

This repository contains the implementation of Single Stage Headless Face detector (https://arxiv.org/abs/1708.03979)

simple_parser.py: Parses the annotation file of WIDER dataset.

ssh_model.py: Contains the SSH model in Keras.

data_generators_gen.py: This is a python data generator which yields X, Y, img_data which is fed into the keras fit_generator

batch_train_ssh.py: This is the python code which is the root for training the model.

config.py: This is the config file which contains all the parameters

data_augment.py: This is used to offset the data to maximize the number of anchors generated

roi_helpers.py: This python file is used to convert the output from the model to bounding boxes

test_ssh_metrics.py: This is the test code which looks at the validation data and returns precision, recall and f-measure
