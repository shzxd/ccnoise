#!/usr/bin/python3
import sys
import caffe
import numpy as np
from scipy.io import loadmat

net_model = 'train/deploy.prototxt' # sys.argv[1]
net_weights = 'train/demo_iter_100000.caffemodel' # sys.argv[2]
phase = 'test'
img_dict = loadmat('data/img.mat')
img = img_dict['img'].astype('float')


def estimate_ccnoise(net_model, net_weights, img):
    net = caffe.Net(net_model, net_weights, caffe.TEST)
    nl = np.zeros((img.shape[0], img.shape[1], 6))
    for i in range(0, img.shape[0], 8):
        for j in range(0, img.shape[1], 8):
            patch = np.transpose(img[i:i+8, j:j+8], [2, 1, 0])
            data = np.concatenate((np.reshape(patch, [3, 64], order='F'), np.tile(np.reshape(patch, [192, 1]), (1, 64))))
            net.blobs['input'].data.shape = [64, 195, 1, 1]
            net.blobs['input'].data[...] = np.reshape(data, [64, 195, 1, 1], order='F')
            out = net.forward()
            out = out['ip2']
            nl[i:i+8, j:j+8, :] = np.transpose(np.reshape(out, [6, 8, 8], order='F'), [2, 1, 0])
    return nl


caffe.set_mode_cpu()
nl = estimate_ccnoise(net_model, net_weights, img)
# print(net.blobs['input'].data)
print(nl)
