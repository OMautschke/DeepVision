import numpy as np

def conv2d(picture, filter):
    convPicture = np.zeros(picture.shape)
    m = int(len(filter)/2)
    for i in range(m, len(picture) - m):
        for j in range(m, len(picture[0]) - m):
            mat = picture[i-m:i+m+1, j-m:j+m+1]
            convPicture[i, j] = np.sum(mat * filter)

    return convPicture

if __name__ == '__main__':
    picture = np.random.rand(5, 5)
    filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    print(conv2d(picture, filter))