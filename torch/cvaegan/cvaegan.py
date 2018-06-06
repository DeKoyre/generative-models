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
Y_dimension = mnist.train.labels.shape[1]
h_dimension = 128
counter = 0
learning_rate = 1e-3
gamma = 1

output_file = open("cgan_output.txt", "w")
text_to_file = ''


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

""" ==================== ENCODER ======================== """

enc_W_xh = xavier_init(size=[X_dimension + Y_dimension, h_dimension])
enc_b_xh = Variable(torch.zeros(h_dimension), requires_grad=True)

enc_W_hz_mu = xavier_init(size=[h_dimension, Z_dimension])
enc_b_hz_mu = Variable(torch.zeros(Z_dimension), requires_grad=True)

enc_W_hz_sigma = xavier_init(size=[h_dimension, Z_dimension])
enc_b_hz_sigma = Variable(torch.zeros(Z_dimension), requires_grad=True)

def sample_z(mu, sigma):
    eps = Variable(torch.randn(batch_size, Z_dimension))
    return mu + torch.exp(sigma / 2) * eps

def Encoder(X, c):
    inputs = torch.cat([X, c], 1)
    h = nn.relu(inputs @ enc_W_xh + enc_b_xh.repeat(inputs.size(0), 1))
    z_mu = h @ enc_W_hz_mu + enc_b_hz_mu.repeat(h.size(0), 1)
    z_sigma = h @ enc_W_hz_sigma + enc_b_hz_sigma.repeat(h.size(0), 1)
    return z_mu, z_sigma



""" ==================== DECODER ======================== """

dec_W_zh = xavier_init(size=[Z_dimension + Y_dimension, h_dimension])
dec_b_zh = Variable(torch.zeros(h_dimension), requires_grad=True)

dec_W_hx = xavier_init(size=[h_dimension, X_dimension])
dec_b_hx = Variable(torch.zeros(X_dimension), requires_grad=True)

def Decoder(z, c):
    inputs = torch.cat([z, c], 1)
    h = nn.relu(inputs @ dec_W_zh + dec_b_zh.repeat(inputs.size(0), 1))
    X = nn.sigmoid(h @ dec_W_hx + dec_b_hx.repeat(h.size(0), 1))
    return X


""" =================== DISCRIMINATOR =================== """

dis_W_hx = xavier_init(size=[X_dimension + Y_dimension, h_dimension])
dis_b_xh = Variable(torch.zeros(h_dimension), requires_grad=True)

dis_W_hd = xavier_init(size=[h_dimension, 1])
dis_b_hd = Variable(torch.zeros(1), requires_grad=True)


def Discriminator(X, c):
    inputs = torch.cat([X, c], 1)
    h = nn.relu(inputs @ dis_W_hx + dis_b_xh.repeat(inputs.size(0), 1))
    d = nn.sigmoid(h @ dis_W_hd + dis_b_hd.repeat(h.size(0), 1))
    return d, h.detach()

enc_params = [enc_W_xh, enc_b_xh, enc_W_hz_mu, enc_b_hz_mu, enc_W_hz_sigma,
              enc_b_hz_sigma]
dec_params = [dec_W_zh, dec_b_zh, dec_W_hx, dec_b_hx]
dis_params = [dis_W_hx, dis_b_xh, dis_W_hd, dis_b_hd]
all_params = enc_params + dec_params + dis_params


""" ===================== TRAINING ======================== """


def reset_grad():
    for p in all_params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())


enc_solver = optim.Adam(enc_params, lr=learning_rate)
dec_solver = optim.Adam(dec_params, lr=learning_rate)
dis_solver = optim.Adam(dis_params, lr=learning_rate)

# ????????????????????????
ones_label = Variable(torch.ones(batch_size, 1))
zeros_label = Variable(torch.zeros(batch_size, 1))

def L_prior_calc(X, X_sample, z_mu, z_sigma):
    e_recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / batch_size
    e_kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_sigma) + z_mu**2 - 1. - z_sigma, 1))
    return e_recon_loss + e_kl_loss

def L_dis_llike_calc(h_real, h_fake):
    nn.sigmoid(h_real)
    return nn.binary_cross_entropy(nn.sigmoid(h_fake), nn.sigmoid(h_real))

def L_gan_calc(D_real, D_fake):
    return torch.sum(torch.log(D_real) + torch.log(torch.sum(1. - D_fake)))

for it in range(100000):
    # обучение энкодера

    X, c = mnist.train.next_batch(batch_size)
    X = Variable(torch.from_numpy(X))
    c = Variable(torch.from_numpy(c.astype('float32')))

    z_mu, z_sigma = Encoder(X, c)
    z = sample_z(z_mu, z_sigma)
    X_sample = Decoder(z, c)

    D_real, h_real = Discriminator(X,c)
    D_fake, h_fake = Discriminator(X_sample, c)

    enc_loss = torch.sum(L_prior_calc(X, X_sample, z_mu, z_sigma) + L_dis_llike_calc(h_real, h_fake))

    enc_loss.backward()
    enc_solver.step()
    reset_grad()









    # обучение декодера
    X, c = mnist.train.next_batch(batch_size)
    X = Variable(torch.from_numpy(X))
    c = Variable(torch.from_numpy(c.astype('float32')))

    z_mu, z_sigma = Encoder(X, c)
    z = sample_z(z_mu, z_sigma)
    X_sample = Decoder(z, c)

    D_real, h_real = Discriminator(X,c)
    D_fake, h_fake = Discriminator(X_sample, c)

    dec_loss = torch.sum(gamma * L_dis_llike_calc(h_real, h_fake) - L_gan_calc(D_real, D_fake))

    dec_loss.backward()
    dec_solver.step()
    reset_grad()









    # обучение дискриминатора

    X, c = mnist.train.next_batch(batch_size)
    X = Variable(torch.from_numpy(X))
    c = Variable(torch.from_numpy(c.astype('float32')))

    z_mu, z_sigma = Encoder(X, c)
    z = sample_z(z_mu, z_sigma)
    X_sample = Decoder(z, c)

    D_real, h_real = Discriminator(X,c)
    D_fake, h_fake = Discriminator(X_sample, c)

    dis_loss = L_gan_calc(D_real, D_fake)
    dis_loss.backward()
    dis_solver.step()
    reset_grad()

    # Print and plot every now and then
    if it % 100 == 0:
        text_to_file = 'i: {}; D_loss: {}; G_loss: {}'.format(it, D_loss.data.numpy(), G_loss.data.numpy())
        output_file.write(text_to_file)
        output_file.flush()

    if it % 1000 == 0:
        print('i: {}; D_loss: {}; G_loss: {}'.format(it, D_loss.data.numpy(), G_loss.data.numpy()))

        c = np.zeros(shape=[batch_size, Y_dimension], dtype='float32')
        c[:, np.random.randint(0, 10)] = 1.
        c = Variable(torch.from_numpy(c))
        samples = G(z, c).data.numpy()[:16]

        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(10, 10)
        # gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(counter).zfill(3)), bbox_inches='tight')
        counter += 1
        plt.close(fig)
