#!/usr/bin/env python
# coding: utf-8

from torchvision import datasets
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from ex1 import convert_mnist_to_vectors, prepare_data
# The first thing to do is specify our model. The perceptron does a matrix
# multiplication of its internal weights with the inputs and adds a bias in
# each layer. After that it activates the resulting vector.
# This can be done using the `Linear` layer. 
# Our model itself will be implemented in an object-oriented manner. You can
# find a skeleton implementation below. Please fill in the blanks marked with
# `# Your code here`. As we want to use mnist vectors as input data, make sure
# to pass the correct dimensions to the `Linear` module. 

# As our perceptron is a pytorch module, it has to inherit from the base
# `nn.Module`.  All pytorch modules expect at least a `forward` method, which
# defines what happens, when you call the instance of such a module on some
# data.


class Perceptron(nn.Module):
    
    def __init__(self, size_hidden=100, size_out=10):
        super().__init__()
        
        self.fc1 = nn.Linear(28*28, size_hidden)
        self.fc2 = nn.Linear(size_hidden, size_hidden)
        self.fc3 = nn.Linear(size_hidden, size_hidden)
        self.fc4 = nn.Linear(size_hidden, size_hidden)
        self.out_layer = nn.Linear(size_hidden, size_out)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # Hint: the content of out at this point is one of the 4 activation
        # features asked for in exercise 3

        # Your Code here: The rest of the layers
        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.relu(out)

        out = self.out_layer(out)

        return out


# Pytorch modules keep track of all model parameters internally. Those will be
# e.g. the matrix and bias of the `Linear` operation we just implemented.

# To be able to feed the mnist vectors to out Perceptron we first have to
# convert them to `torch.Tensor`s. To not have to do this everytime we want to
# do an operation on those vectors you can find a `torch.Dataset` version of
# the mnist vectors below. All it does is a simple casting operation.

class MnistVectors(torch.utils.data.Dataset):
    
    '''A Pytorch Dataset, which does the same data preparation as was done in
    the PCA exercise.'''

    def __init__(self, split='train'):
        super().__init__()
        
        mnist = datasets.MNIST('../data',
                               train=split=='train',
                               download=True)

        ########################
        #### Your Code here ####
        self.mnist_vectors, self.labels = convert_mnist_to_vectors(mnist)
        self.mnist_vectors = prepare_data(self.mnist_vectors)
        ########################

            
    def __getitem__(self, idx):
        '''Implements the ``[idx]`` method. Here we convert the numpy data to
        torch tensors.
        The images are needed as Variables, so that we can differenciate the
        graph, we later generate when applying our model to theses images.
        '''

        mvec = torch.autograd.Variable(
                    torch.tensor(self.mnist_vectors[idx]).float(),
                    requires_grad=True
                    )
        label = torch.tensor(self.labels[idx]).long()

        return mvec, label
    
    def __len__(self):
        return len(self.labels)


# The following two functions are needed to track the progress of the training.
# One transforms the output of the Perceptron into a skalar class label, the 
# other uses that label to calculate the batch accuracy.

def batch_accuracy(prediction, label):
    return np.sum(np.where(label == prediction, 1, 0)) / len(prediction)

def class_label(prediction):
    return torch.argmax(prediction, dim=1)

def train(use_gpu=False):
    
    # Here we instanciate our model. The weights of the model are automatically
    # initialized by pytorch
    P = Perceptron()
    
    TrainData = MnistVectors()
    TestData = MnistVectors('test')
    # Dataloaders allow us to load the data in batches. This allows us a better
    # estimate of the parameter updates when doing backprop.
    # We need two Dataloaders so that we can train on the train data split
    # and evaluate on the test datasplit.
    Dl = DataLoader(TrainData, batch_size=16, shuffle=True)
    testDl = DataLoader(TestData, batch_size=16, shuffle=False)
    
    # Use the Adam optimizer with learning rate 1e-4 and otherwise default
    # values
    optimizer =  torch.optim.Adam(P.parameters(), lr=1e-4)

    # Use the Cross Entropy loss from pytorch. Make sure your Perceptron does
    # not use any activation function on the output layer!
    criterion = nn.CrossEntropyLoss()
    
    if use_gpu:
        P.cuda()
        criterion.cuda()
    
    for epoch in tqdm(range(5), desc='Epoch'):
        for step, [example, label] in enumerate(tqdm(Dl, desc='Batch')):
            if use_gpu:
                example = example.cuda()
                label = label.cuda()
            
            # The optimizer knows about all model parameters. These in turn
            # store their own gradients. When calling loss.backward() the newly
            # computed gradients are added on top of the existing ones. Thus
            # at before calculating new gradients we need to clear the old ones
            # using ther zero_grad() method.
            optimizer.zero_grad()
            
            prediction = P(example)
            
            loss = criterion(prediction, label)
            
            # Here pytorch applies backpropagation for us completely
            # automatically!!! That is quite awsome!
            loss.backward()

            # The step method now adds the gradients onto the model parameters
            # as specified by the optimizer and the learning rate.
            optimizer.step()
            
            # To keep track of what is happening print some outputs from time
            # to time.
            if (step % 375) == 0:
                # Your code here
                acc = batch_accuracy(class_label(prediction), label)
                tqdm.write('Batch Accuracy: {}%, Loss: {}'.format(acc, loss))
                
        # Now validate on the whole test set
        accuracies = []
        for idx, [test_ex, test_l] in enumerate(tqdm(testDl, desc='Test')):
            if use_gpu:
                test_ex = test_ex.cuda()
                test_l = test_l.cuda()

            #########################
            #### Your Code here  ####
            #########################

            # Using your batch_accuracy function, also print the mean accuracy
            # over the whole test split of the data.

        print('Validation Accuracy: {}%'.format(np.mean(accuracies)))

        # Now let's write out a checkpoint of the model, so that we can
        # reuse it:
        torch.save(model.state_dict(), 'perceptron_{}.ckpt'.format(step))

        # If you need to load the checkpoint instanciate your model and the
        # load the state dict from a checkpoint:
        # P = Perceptron()
        # P.load_state_dict(torch.load(perceptron_3750.ckpt))
        # Make sure to use the latest checkpoint by entering the right number.


if __name__ == '__main__':
    train()
