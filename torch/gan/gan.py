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
import time

start_time = time.time()

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
batch_size = 64
z_dimension = 100
x_dimension = mnist.train.images.shape[1]
d_dimension = mnist.train.labels.shape[1]
h_dimension = 128
iterator = 0
learning_rate = 1e-3

output_file = open("gan_output.txt", "w")
text_to_file = ''


def xavier_init(size):
    input_dimension = size[0]
    xavier_stddev = 1. / np.sqrt(input_dimension / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


""" ==================== GENERATOR ======================== """

w_zh = xavier_init(size=[z_dimension, h_dimension])
b_zh = torch.zeros(h_dimension, requires_grad=True)

w_hx = xavier_init(size=[h_dimension, x_dimension])
b_hx = torch.zeros(x_dimension, requires_grad=True)


def G(z):
    h = nn.relu(z @ w_zh + b_zh.repeat(z.size(0), 1))
    X = nn.sigmoid(h @ w_hx + b_hx.repeat(h.size(0), 1))
    return X


""" ==================== DISCRIMINATOR ======================== """

w_xh = xavier_init(size=[x_dimension, h_dimension])
b_xh = Variable(torch.zeros(h_dimension), requires_grad=True)

w_hp = xavier_init(size=[h_dimension, 1])
b_hp = Variable(torch.zeros(1), requires_grad=True)


def D(X):
    h = nn.relu(X @ w_xh + b_xh.repeat(X.size(0), 1))
    y = nn.sigmoid(h @ w_hp + b_hp.repeat(h.size(0), 1))
    return y


G_params = [w_zh, b_zh, w_hx, b_hx]
D_params = [w_xh, b_xh, w_hp, b_hp]
params = G_params + D_params


""" ===================== TRAINING ======================== """


def reset_grad():
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())


G_solver = optim.Adam(G_params, lr=learning_rate)
D_solver = optim.Adam(D_params, lr=learning_rate)

ones_label = Variable(torch.ones(batch_size, 1))
zeros_label = Variable(torch.zeros(batch_size, 1))


for it in range(100000):
    # Sample data
    z = Variable(torch.randn(batch_size, z_dimension))
    X, _ = mnist.train.next_batch(batch_size)
    X = Variable(torch.from_numpy(X))

    # Подсчет ошибки для дискриминатора и шаг обратного распространения
    G_sample = G(z)
    D_real = D(X)
    D_fake = D(G_sample)

    D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
    D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label)
    D_loss = D_loss_real + D_loss_fake

    D_loss.backward()
    D_solver.step()

    reset_grad()

    # Подсчет ошибки для генератора и шаг обратного распространения
    z = Variable(torch.randn(batch_size, z_dimension))
    G_sample = G(z)
    D_fake = D(G_sample)

    G_loss = nn.binary_cross_entropy(D_fake, ones_label)

    G_loss.backward()
    G_solver.step()

    reset_grad()

    if it % 100 == 0:
        text_to_file = 'i: {}; D_loss: {:.4}; G_loss: {:.4}; sec: {:.4}\n'.format(it, D_loss.data.numpy(), G_loss.data.numpy(), (time.time() - start_time))
        output_file.write(text_to_file)
        output_file.flush()

    # На каждой 1000 итерации денмонстрация результатов
    if it % 1000 == 0:
        print('i: {}; D_loss: {:.4}; G_loss: {:.4}; sec: {};\n'.format(it, D_loss.data.numpy(), G_loss.data.numpy(), (time.time() - start_time)))

        samples = G(z).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(iterator).zfill(3)), bbox_inches='tight')
        iterator += 1
        plt.close(fig)
