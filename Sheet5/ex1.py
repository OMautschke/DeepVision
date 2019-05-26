import matplotlib.pyplot as plt
import numpy as np


def norm0_x1(x):
    return np.ones_like(x) * 6

def norm0_x2(x):
    return np.ones_like(x) * 3

def f(x):
    return 6 - 2 * x

def norm1(x):
    return abs(f(x)) + abs(x)

def norm2(x):
    return np.sqrt(x**2 + (f(x))**2)

def det(x):
    return (5 * x - 12) / (np.sqrt(36 - 24 * x + 5 * x**2))


A = np.arange(-6, 6, 0.5)

fig = plt.figure()
a1 = fig.add_subplot(2, 2, 1)
a1.plot(A, norm0_x1(A), A, norm0_x2(A), A, f(A))
a1.set_xlabel("x1")
a1.set_ylabel("x2")
a1.legend(("x1 = 0", "x2 = 0", "x1 + 2x2 = 6"))


# Norm 1
a2 = fig.add_subplot(2, 2, 2)
a2.plot(A, norm1(A))
a2.set_xlabel("x1")
a2.set_ylabel("y")

normMin = np.min(norm1(A))

a2.scatter(normMin, norm1(normMin), c="r")
a2.legend(("norm1", "min"))
a2.set_title("Norm1 and it's min")


# Norm 2
a2 = fig.add_subplot(2, 2, 3)
a2.plot(A, norm2(A))
a2.set_xlabel("x")
a2.set_ylabel("y")

normMin = np.min(norm2(A))

a2.scatter(normMin, norm2(normMin), c="r")
a2.legend(("norm2", "min"))
a2.set_title("Norm2 and it's min")

# Norm 2 det
a2 = fig.add_subplot(2, 2, 4)
a2.plot(A, det(A))
a2.set_xlabel("x")
a2.set_ylabel("y")

normMin = np.min(det(A))

a2.scatter(normMin, det(normMin), c="r")
a2.legend(("norm2", "min"))
a2.set_title("Norm2 extended")

plt.show()