#!/usr/bin/env python
# coding: utf-8

from torchvision import datasets
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm  # Not needed but very cool!


def load_data(train=True):
    mnist = datasets.MNIST('../data',
                           train=train,
                           download=False)
    return mnist


def plot_examples(data):
    #########################
    #### Your Code here  ####
    #########################
    # Plot some example and put their corresponding label on top as title.

    # Also print some statistics
    for i in range(10):
        minimum = torch.min(data.data[i]).item()
        maximum = torch.max(data.data[i]).item()
        mean    = torch.mean(data.data[i].float()).item()
        shape   = 1#data.data[i].size()
        dtype   = "type"#data.data[i].dtype
        print("Image " + str(i) + " --- Min:" + str(minimum) + " Max:" + str(maximum) +
                " Mean:" + str(mean) + " Shape:" + str(shape) + " DType:" + dtype)
    pass


def convert_mnist_to_vectors(data):
    '''Converts the ``[28, 28]`` MNIST images to vectors of size ``[28*28]``.
    '''

    mnist_vectors = []
    labels = []

    for image, label in tqdm(data):
        #########################
        #### Your Code here  ####
        #########################        
        mnist_vectors.append(np.array(image).reshape((1, 28*28)))
        labels.append(label)
        pass
        


    return mnist_vectors, labels


def prepare_data(data):
    '''Centers the data around 0 and rescales it to the range of ``[-1, 1]``.
    '''

    #########################
    #### Your Code here  ####
    #########################
    for vec in data:
        print((vec - 128) / 128)

    return data


def do_pca(data):

    data = load_data()

    mnist_vectors, labels = convert_mnist_to_vectors(data)
    prepare_data(mnist_vectors)

    #########################
    #### Your Code here  ####
    #     
    # cov = ...
    # eigenVectors, eigenValues = torch.eig(data.data[0])

    # sort eigenVectors by eigenValues
    #########################

    return 1#sorted_eigenVectors


def plot_pcs(sorted_eigenVectors, num=10):
    '''Plots the first ``num`` eigenVectors as images.'''

    #########################
    #### Your Code here  ####
    #     
    # Reshape
    # vec_as_im = ...

    # Plot
    # ...
    #########################


def plot_projection(sorted_eigenVectors, data):
    '''Projects ``data`` onto the first two ``sorted_eigenVectors`` and makes
    a scatterplot of the resulting points'''

    #########################
    #### Your Code here  ####
    #     
    # Projection
    # data_on_pcs = dot-product of data and first two eigen vectors

    # Plot
    # ...
    #########################



if __name__ == '__main__':
    # You can run this part of the code from the terminal using python ex1.py

    data = load_data()

    plot_examples(data)
    pcs = do_pca(data)

    plot_pcs(pcs)
    plot_projection(pcs, data)
