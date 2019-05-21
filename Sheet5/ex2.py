import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as mtp

def calc_fourier(f, X):
    Y = f(X)
    return fft.fft(Y)

def calc_fourier_inc_res(f, X, m):
    Y = f(X)
    YF = fft.fft(Y)
    SYF = fft.fftshift(YF)

    return np.pad(SYF, int(m/2), 'constant')

def transform_back(YF, n, m):
    Y = fft.ifft(fft.ifftshift(YF))
    return Y.real * ((n + m) / n)

if __name__ == '__main__':
    n = 100
    m = 10
    X = np.linspace(-10, 10, n)

    f = lambda x: np.exp(np.power(-x, 2))
    Y    = f(X)
    YF   = calc_fourier(f, X)
    YB = fft.ifft(YF)

    mtp.subplot(1, 3, 1)
    mtp.plot(X, Y.real)
    mtp.title('Original')
    mtp.subplot(1, 3, 2)
    mtp.plot(YF.real)
    mtp.title('Fourier')
    mtp.subplot(1, 3, 3)
    mtp.plot(YB.real)
    mtp.title('Back Transformation')
    mtp.show()

    YF = calc_fourier_inc_res(f, X, 10)
    YB = transform_back(YF, n, m)

    mtp.subplot(1, 3, 1)
    mtp.plot(X, Y.real)
    mtp.title('Original')
    mtp.subplot(1, 3, 2)
    mtp.plot(YF.real)
    mtp.title('Fourier Shifted')
    mtp.subplot(1, 3, 3)
    mtp.plot(YB.real)
    mtp.title('Back Transformation HR')

    mtp.show()

    f = lambda x: 0 if x < -5 or x > 5 else 1
    f = np.vectorize(f)
    Y    = f(X)
    YF = calc_fourier_inc_res(f, X, 10)
    YB = transform_back(YF, n, m)

    mtp.subplot(1, 3, 1)
    mtp.plot(X, Y.real)
    mtp.title('Original')
    mtp.subplot(1, 3, 2)
    mtp.plot(YF.real)
    mtp.title('Fourier Shifted')
    mtp.subplot(1, 3, 3)
    mtp.plot(YB.real)
    mtp.title('Back Transformation HR with Artefacts')
    mtp.show()