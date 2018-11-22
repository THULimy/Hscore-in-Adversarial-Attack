# -*- coding: utf-8 -*-
# @Author: limy
# @Date:   2018-07-27 11:40:19
# @Last Modified by:   limy
# @Last Modified time: 2018-07-27 16:41:52
import tensorflow as tf
import numpy as np
import pickle
import glob

def unpickle(file):
	#unpickle
	with open(file,'rb') as fo:
		dict=pickle.load(fo,encoding='bytes')
	return dict

def preprocessing(images):
	images=images/255.
	images=images-images.mean(axis=0)
	images=np.moveaxis(images,1,-1)
	return images

# def convert()

def loadcifar():
	filedir='/home/limy/Dataset/cifar-10-batches-py/'
	raw_images=[]
	labels=[]
	#load data
	for file in glob.glob(filedir+'data_batch*'):
		temp=unpickle(file)
		raw_images.append(temp[b'data'])
		labels.append(temp[b'labels'])
	#reshape
	raw_images=np.array(raw_images).reshape(-1,3,32,32)
	labels=np.array(labels).flatten()
	#preprocess
	images=preprocessing(raw_images)
	return images,labels

def VGGblock(inputs,blocknum,layernum,filternum):
	for num in range(layernum):
		layername='conv'+num2str(blocknum)+num2str(num+1)
		if num==0:
			net=tf.layers.conv2d(inputs=inputs,filters=filternum,name=layername,kernel_size=3,padding='same',activation=tf.nn.relu)
		else:
			net=tf.layers.conv2d(inputs=net,filters=filternum,name=layername,kernel_size=3,padding='same',activation=tf.nn.relu)
	net=tf.layers.max_pooling2d(inputs=net,pool_size=2,stride=2)
	return net

def VGG16(inputs):
	net=VGGblock(input=inputs,blocknum=1,layernum=2,filternum=64)
	net=VGGblock(input=net,blocknum=2,layernum=2,filternum=128)
	net=VGGblock(input=net,blocknum=3,layernum=3,filternum=256)
	net=VGGblock(input=net,blocknum=4,layernum=3,filternum=512)
	net=VGGblock(input=net,blocknum=5,layernum=3,filternum=512)
	net=tf.layers.flatten(net)
	net=tf.layers.dense(inputs=net,name='fc1',units=4096,activation=tf.nn.relu)
	net=tf.layers.dense(inputs=net,name='fc2',units=4096,activation=tf.nn.relu)
	net=tf.layers.dense(inputs=net,name='fc3',units=1000,activation=tf.nn.relu)
	return net


images,labels=loadcifar()
print(images[0].shape)

