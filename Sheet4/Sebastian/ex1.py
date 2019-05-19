#!/usr/bin/env python
# coding: utf-8

from torchvision import datasets
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from tqdm.auto import tqdm  # Not needed but very cool!


def load_data(train=True):
    mnist = datasets.MNIST('../data',
                           train=train,
                           download=True)
    return mnist


def plot_examples(data):
    #########################
    #### Your Code here  ####
    #########################

    # Plot some examples and put their corresponding label on top as title.
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("1")
    ax1.imshow(data.data[0])

    ax1 = fig.add_subplot(2, 2, 2)
    ax1.set_title("2")
    ax1.imshow(data.data[1])

    ax1 = fig.add_subplot(2, 2, 3)
    ax1.set_title("3")
    ax1.imshow(data.data[2])

    ax1 = fig.add_subplot(2, 2, 4)
    ax1.set_title("4")
    ax1.imshow(data.data[3])
    plt.show()


    # Also print some statistics
    for i in range(10):
        minimum = torch.min(data.data[i]).item()
        maximum = torch.max(data.data[i]).item()
        mean = torch.mean(data.data[i].float()).item()
        shape = data.data[i].shape
        dtype = data.data[i].dtype
        print("Image ", str(i), " --- Min:", str(minimum), " Max:", str(maximum),
              " Mean:", str(mean), " Shape:", str(shape), " DType:", dtype)


def convert_mnist_to_vectors(data):
    '''Converts the ``[28, 28]`` MNIST images to vectors of size ``[28*28]``.
    '''

    mnist_vectors = []
    labels = []

    for image, label in tqdm(data):
        #########################
        #### Your Code here  ####
        #########################
        mnist_vectors.append(np.array(image).reshape((28 * 28)))
        labels.append(label)

    return mnist_vectors, labels


def prepare_data(data):
    '''Centers the data around 0 and rescales it to the range of ``[-1, 1]``.
    '''

    #########################
    #### Your Code here  ####
    #########################
    images = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):
        images[i] = np.interp(data[i], (data[i].min(), data[i].max()), (-1, +1))
    return images


def do_pca(data):
    mnist_vectors, labels = convert_mnist_to_vectors(data)
    scaled_data = prepare_data(mnist_vectors)

    #########################
    #### Your Code here  ####

    #cov = np.cov(scaled_data)

    eigVal, eigVec= np.linalg.eig(np.reshape(scaled_data, (60000, 28, 28)))

    # sort eigenVectors by eigenValues
    #idx = np.argsort(eigVal)[::-1]
    #eigVec = eigVec[:, idx]

    return eigVec


def plot_pcs(sorted_eigenVectors, num=10):
    '''Plots the first ``num`` eigenVectors as images.'''

    #########################
    #### Your Code here  ####
    #     
    # Reshape
    vec_as_im = np.reshape(sorted_eigenVectors[0:num, :], (num, 28, 28))

    # Plot
    fig = plt.figure()
    for i in range(num):
        a = fig.add_subplot(np.ceil(num / 3),
                            np.ceil(num / 3),
                            i + 1)
        a.imshow(vec_as_im[i].real)
    plt.show()


def plot_projection(sorted_eigenVectors, data):
    '''Projects ``data`` onto the first two ``sorted_eigenVectors`` and makes
    a scatterplot of the resulting points'''

    mnist_vectors, labels = convert_mnist_to_vectors(data)
    scaled_data = prepare_data(mnist_vectors)

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

    #plot_examples(data)
    pcs = do_pca(data)

    #plot_pcs(pcs)
    plot_projection(pcs, data)
