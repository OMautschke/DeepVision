import numpy as np

class Linear(object):
    @staticmethod
    def forward(W, b, x):
        # TODO implement Wx + b with support for batches
        # W are weights    (n_out, n_in)
        # b are biases     (n_out,)
        # x are inputs     (batch_size, n_in)
        # output should be (batch_size, n_out)
        return W.dot(x) + b

    @staticmethod
    def backward(W, b, x, dout):
        dout_ = dout[:,:, None]
        W_ = W[None,:,:]
        dx = dout_*W_
        dx = np.sum(dx, axis = 1)
        # TODO implement dW and db
        return dW, db, dx


class ReLU(object):
    @staticmethod
    def forward(x):
        # TODO implement ReLU
        return

    @staticmethod
    def backward(x, dout):
        # TODO implement dx
        return (dx,)


class CELoss(object):
    @staticmethod
    def forward(x, y):
        assert len(x.shape) == 2 # x is batch of predictions   (batch_size, 10)
        assert len(y.shape) == 1 # y is batch of target labels (batch_size,)
        # TODO implement cross entropy loss averaged over batch
        return


    @staticmethod
    def backward(x, y, dout):
        # TODO implement dx
        dy = 0.0 # no useful gradient for y, just set it to zero
        return dx, dy


def get_W(n_in, n_out):
    # he init for weights of Linear layer
    bound = 1.0 / np.sqrt(n_in)
    return np.random.uniform(low = -bound, high = bound, size = [n_out, n_in])


def get_b(n_in, n_out):
    # he init for bias of Linear layer
    bound = 1.0 / np.sqrt(n_in)
    return np.random.uniform(low = -bound, high = bound, size = [n_out])


class MLP(object):
    def __init__(self, size_hidden = 100, size_out = 10):
        self.verbose = verbose

        values = dict()
        values["W0"] = get_W(28*28, size_hidden)
        values["b0"] = get_b(28*28, size_hidden)

        values["W1"] = get_W(size_hidden, size_hidden)
        values["b1"] = get_b(size_hidden, size_hidden)
        values["W2"] = get_W(size_hidden, size_hidden)
        values["b2"] = get_b(size_hidden, size_hidden)
        values["W3"] = get_W(size_hidden, size_hidden)
        values["b3"] = get_b(size_hidden, size_hidden)

        values["W4"] = get_W(size_hidden, size_out)
        values["b4"] = get_b(size_hidden, size_out)

        self.values = values
        self.trainable_variables = sorted(values.keys())

        self.tape = list()
        self.tape.append(("z0", Linear, ("W0", "b0", "x")))
        # TODO complete the tape for z0,...,z8
        self.tape.append(("z9", CELoss, ("z8", "y")))

        self.endpoints = [node[0] for node in self.tape]


    def forward(self, x, y):
        # init input values
        self.values["x"] = x
        self.values["y"] = y
        for (node, op, inputs) in self.tape:
            # TODO forward traverse and evaluate tape

        return dict((k, self.values[k]) for k in self.endpoints)


    def backward(self):
        # init all adjoints with zero
        self.deltas = dict()
        for k in self.values:
            self.deltas[k] = 0.0
        # init the output adjoint with one
        self.deltas["z9"] = np.array(1.0) # use array to have .shape on all deltas
        # traverse the tape in reverse to compute adjoints of all nodes
        for (node, op, inputs) in reversed(self.tape):
            dinputs = # TODO compute adjoints of all parents

            # distribute adjoints to parents
            for input_, dinput in zip(inputs, dinputs):
                self.deltas[input_] += dinput
        return self.deltas


    def step(self, learning_rate = 1e-1):
        for k in self.trainable_variables:
            self.values[k] = self.values[k] - learning_rate*self.deltas[k]


def check_gradients(mlp, batch):
    x, y = batch
    x = x.reshape([-1,28*28])

    from pt_mlp import PT_MLP, torch
    pt_mlp = PT_MLP()
    # TODO copy weights and biases from mlp to pt_mlp

    pt_endpoints = pt_mlp(x, y)
    # must explicitly tell pytorch to retain gradients of intermediate tensors
    for k in pt_endpoints:
        pt_endpoints[k].retain_grad()
    pt_endpoints["z9"].backward()
    pt_grads = dict((k, pt_endpoints[k].grad) for k in pt_endpoints)

    np_endpoints = mlp.forward(x.numpy(), y.numpy())
    np_grads = mlp.backward()

    for k in sorted(pt_grads.keys()):
        npgrad = np_grads[k]
        ptgrad = pt_grads[k].numpy()
        print("np", k, npgrad.shape)
        print("pt", k, ptgrad.shape)
        print("error", np.linalg.norm(npgrad-ptgrad))


def train(mlp, batches):
    for batch in batches:
        x, y = batch
        x = x.numpy()
        y = y.numpy()
        x = x.reshape([-1,28*28])

        endpoints = mlp.forward(x, y)
        loss = endpoints["z9"]
        print(loss)

        deltas = mlp.backward()
        mlp.step()


mlp = MLP()
from data import get_batches
batches = get_batches(train = True, batch_size = 32, shuffle = True)
check_gradients(mlp, next(iter(batches)))
train(mlp, batches)
