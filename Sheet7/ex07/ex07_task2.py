TOL = 1e-4

def f(x0, x1):
    return (2*(x0+x1)+(x0+x1)**2)**2# TODO expression for f

def gradf(x0, x1):
    d0 = 4*(2*(x0+x1)+3*(x0+x1)**2+(x0+x1)**3)# TODO partial derivative wrt x0
    d1 = d0# TODO partial derivative wrt x1
    return d0, d1

print("f(1,2) = ", f(1, 2))
print("gradf(1,2) = ", gradf(1, 2))

# numerical check of gradient
eps = 1e-8
approxerr0 = abs(((f(1+eps,2) - f(1,2)) / eps) - gradf(1, 2)[0])
approxerr1 = abs(((f(1,2+eps) - f(1,2)) / eps) - gradf(1, 2)[1])
assert approxerr0 < TOL, approxerr0
assert approxerr1 < TOL, approxerr1

# implementations of the elementary functions used for autodiff
# forward implements the computation of the function
# backward implements the computation of the adjoints of each input
# backward should return a tuple with as many elements as the function has
# inputs, and the i-th element should be
# dout*(partial derivative with respect to the i-th input)
class plus(object):
    @staticmethod
    def forward(x, y):
        return x+y

    @staticmethod
    def backward(x, y, dout):
        dx = 1.0
        dy = 1.0
        return dout*dx, dout*dy

class double(object):
    @staticmethod
    def forward(x):
        return 2*x

    @staticmethod
    def backward(x, dout):
        return (2*dout, )

class square(object):
    @staticmethod
    def forward(x):
        return x**2

    @staticmethod
    def backward(x, dout):
        return (2*x*dout, )

# trace the computation of f in terms of the elementary functions on a tape
tape = list()
tape.append(("z0", plus, ("x0", "x1")))
tape.append(("z1", double, ("z0",)))
tape.append(("z2", square, ("z0",)))
tape.append(("z3", plus, ("z1", "z2")))
tape.append(("z4", square, ("z3",)))

# TODO finish computation until "z4": append tuples of length three containing
# - name of node to be added
# - elementary function to apply
# - tuple containing names of the input nodes

def forward(x0, x1):
    # store values for all nodes in a dictionary
    values = dict()
    # init input values
    values["x0"] = x0
    values["x1"] = x1
    # traverse the tape to compute values of all nodes
    for (node, op, inputs) in tape:
        values[node] = op.forward(*[values[inpt] for inpt in inputs])# TODO compute value of node using op.forward
    return values

def backward(values):
    # store adjoints of all nodes in a dictionary
    deltas = dict()
    # init all adjoints with zero
    for k in values:
        deltas[k] = 0.0
    # init the output adjoint with one
    deltas["z4"] = 1.0
    # traverse the tape in reverse to compute adjoints of all nodes
    for (node, op, inputs) in reversed(tape):
        dinputs = op.backward(*[values[inpt] for inpt in inputs], deltas[node])# TODO compute adjoints of all parents
        # distribute adjoints to parents
        for input_, dinput in zip(inputs, dinputs):
            deltas[input_] += dinput
    return deltas

values = forward(1, 2)
err = abs(values["z4"] - f(1,2))
assert err < TOL, err
deltas = backward(values)
print(values)
print(deltas)
err0 = abs(deltas["x0"] - gradf(1,2)[0])
err1 = abs(deltas["x1"] - gradf(1,2)[1])
assert err0 < TOL, err0
assert err1 < TOL, err1
