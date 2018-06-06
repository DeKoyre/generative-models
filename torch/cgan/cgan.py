import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
batch_size = 64
Z_dimension = 100
X_dimension = mnist.train.images.shape[1]
num_classes = mnist.train.labels.shape[1]
h_dimension = 128
counter = 0
learning_rate = 1e-3

output_file = open("cgan_output.txt", "w")
text_to_file = ''


def xaviers_initialization(size):
    input_dimension = size[0]
    xavier_standart_deviation = 1. / np.sqrt(input_dimension / 2.)
    return Variable(torch.randn(*size) * xavier_standart_deviation, requires_grad=True)


""" ==================== ГЕНЕРАТОР ======================== """

W_zh = xaviers_initialization(size=[Z_dimension + num_classes, h_dimension])
b_zh = Variable(torch.zeros(h_dimension), requires_grad=True)

W_hx = xaviers_initialization(size=[h_dimension, X_dimension])
b_hx = Variable(torch.zeros(X_dimension), requires_grad=True)


def G(z, c):
    inputs = torch.cat([z, c], 1)
    h = nn.relu(inputs @ W_zh + b_zh.repeat(inputs.size(0), 1))
    X = nn.sigmoid(h @ W_hx + b_hx.repeat(h.size(0), 1))
    return X


""" ==================== ДИСКРИМИНАТОР ======================== """

W_xh = xaviers_initialization(size=[X_dimension + num_classes, h_dimension])
b_xh = Variable(torch.zeros(h_dimension), requires_grad=True)

W_hp = xaviers_initialization(size=[h_dimension, 1])
b_hp = Variable(torch.zeros(1), requires_grad=True)


def D(X, c):
    inputs = torch.cat([X, c], 1)
    h = nn.relu(inputs @ W_xh + b_xh.repeat(inputs.size(0), 1))
    p = nn.sigmoid(h @ W_hp + b_hp.repeat(h.size(0), 1))
    return p


G_params = [W_zh, b_zh, W_hx, b_hx]
D_params = [W_xh, b_xh, W_hp, b_hp]
params = G_params + D_params


""" ===================== ОБУЧЕНИЕ ======================== """


def reset_grad():
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())


G_solver = optim.Adam(G_params, lr=learning_rate)
D_solver = optim.Adam(D_params, lr=learning_rate)

ones_label = Variable(torch.ones(batch_size, 1))
zeros_label = Variable(torch.zeros(batch_size, 1))

discriminator_steps = 3

for it in range(200000):
    z = Variable(torch.randn(batch_size, Z_dimension))  # генеруем z
    X, c = mnist.train.next_batch(batch_size)           # берем батч
    X = Variable(torch.from_numpy(X))                   # переводим X в тензор
    c = Variable(torch.from_numpy(c.astype('float32'))) # переводим с в тензор

    for i in range(discriminator_steps):
        # Делаем шаг обратного распространения для обновления значения дискриминатора
        G_sample = G(z, c)
        D_real = D(X, c)
        D_fake = D(G_sample, c)

        D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
        D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake

        D_loss.backward()
        D_solver.step()

        # Сбрасываем градиент
        reset_grad()

    # Делаем шаг обратного распространения для обновления значения генератора
    z = Variable(torch.randn(batch_size, Z_dimension))
    G_sample = G(z, c)
    D_fake = D(G_sample, c)

    G_loss = nn.binary_cross_entropy(D_fake, ones_label)

    G_loss.backward()
    G_solver.step()

    # Сбрасываем градиент
    reset_grad()

    # Print and plot every now and then
    if it % 100 == 0:
        text_to_file = 'i: {}; D_loss: {}; G_loss: {}\n'.format(it, D_loss.data.numpy(), G_loss.data.numpy())
        output_file.write(text_to_file)
        output_file.flush()

    if it % 1000 == 0:
        print('I: {}; D_loss: {}; G_loss: {}'.format(it, D_loss.data.numpy(), G_loss.data.numpy()))

        for label in range(num_classes):
            c = np.zeros(shape=[batch_size, num_classes], dtype='float32')
            c[:, label] = 1.
            c = Variable(torch.from_numpy(c))
            samples = G(z, c).data.numpy()[:25]

            fig = plt.figure(figsize=(5, 5))
            gs = gridspec.GridSpec(5, 5)

            for i, sample in enumerate(samples):
                sample = 1 - sample
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

            if not os.path.exists('out/'):
                os.makedirs('out/')

            plt.savefig('out/{}-{}.png'.format(label, str(it)), bbox_inches='tight')
            counter += 1
            plt.close(fig)
