#!/usr/bin/python3
import sys
import caffe
import numpy as np
from scipy.io import loadmat

net_model = sys.argv[1]
net_weights = sys.argv[2]
img_dict = loadmat(sys.argv[3])
img = img_dict['img'].astype('float')


def estimate_ccnoise(net_model, net_weights, img):
    net = caffe.Net(net_model, net_weights, 'test')
    nl = np.zeros((img.shape[0]), img.shape[1], 6)
    for i in range(0, img.shape[0], 8):
        for j in range(0, img.shape[1], 8):
            patch = np.transpose(img[i:i+8, j:j+8], [2, 1, 0])
            data = np.concatenate((np.reshape(patch, [3, 64], order='F'), np.tile(np.reshape(patch, [192, 1]), (1, 64))))
            net.blobs['data'].data.shape = [1, 1, 195, 64]
            net.blobs['data'].data[...] = np.reshape(data, [1, 1, 195, 64], order='F')
            out = net.forward()
            out = out[1]
            nl[i:i+8, j:j+8, :] = np.transpose(np.reshape(out, [6, 8, 8], order='F'), [2, 1, 0])


caffe.set_mode_cpu()
nl = estimate_ccnoise(net_model, net_weights, img)
caffe.reset_all()
