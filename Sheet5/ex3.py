import numpy as np
import matplotlib.pyplot as plt

def conv2d(picture, filter):
    convPicture = np.zeros(picture.shape)
    m = int(len(filter)/2)
    for i in range(m, len(picture) - m):
        for j in range(m, len(picture[0]) - m):
            mat = picture[i-m:i+m+1, j-m:j+m+1]
            convPicture[i, j] = np.sum(mat * filter)

    return convPicture

if __name__ == '__main__':
    filterA = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    filterB = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    filterC = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    pic = plt.imread('cameraman.png')
    org = np.array(pic)

    fA = conv2d(org, filterA)
    fB = conv2d(org, filterB)
    fC = conv2d(org, filterC)

    plt.subplot(2, 2, 1)
    plt.imshow(org, 'gray')
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(fA, 'gray')
    plt.title('Filter A')
    plt.subplot(2, 2, 3)
    plt.imshow(fB, 'gray')
    plt.title('Filter B')
    plt.subplot(2, 2, 4)
    plt.imshow(fC, 'gray')
    plt.title('Filter C')
    plt.show()