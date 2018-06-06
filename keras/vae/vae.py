from keras.callbacks import LambdaCallback, ReduceLROnPlateau, TensorBoard
from keras.datasets import fashion_mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input, Lambda, Reshape
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras import backend as K

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test  = np.reshape(x_test,  (len(x_test),  28, 28, 1))

batch_size = 500
Z_dimension = 2
dropout_rate = 0.3
start_lr = 0.0001

path_to_img = "images/"
directory = os.path.dirname(path_to_img)
if not os.path.exists(directory):
    os.makedirs(directory)

text_file = open("vae_output.txt", "w")
text_to_file = ''

def create_vae():
    models = {}

    """ ======= ЭНКОДЕР ======= """
    inputLayer = Input(batch_shape=(batch_size, 28, 28, 1))
    x = Flatten()(inputLayer)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    z_mu = Dense(Z_dimension)(x)
    z_sigma = Dense(Z_dimension)(x)

    def sampling(args):
        z_mu, z_sigma = args
        epsilon = K.random_normal(shape=(batch_size, Z_dimension), mean=0., stddev=1.0)
        return z_mu + K.exp(z_sigma / 2) * epsilon
    l = Lambda(sampling, output_shape=(Z_dimension,))([z_mu, z_sigma])

    models["encoder"]  = Model(inputLayer, l, 'Encoder')
    models["z_mu"] = Model(inputLayer, z_mu, 'Enc_z_mu')
    models["z_sigma"] = Model(inputLayer, z_sigma, 'Enc_z_sigma')

    """ ======= ДЕКОДЕР ======= """
    z = Input(shape=(Z_dimension, ))
    x = Dense(128)(z)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(28*28, activation='sigmoid')(x)
    decoded = Reshape((28, 28, 1))(x)

    models["decoder"] = Model(z, decoded, name='Decoder')
    models["vae"]     = Model(inputLayer, models["decoder"](models["encoder"](inputLayer)), name="VAE")

    def vae_loss(x, decoded):
        x = K.reshape(x, shape=(batch_size, 28*28))
        decoded = K.reshape(decoded, shape=(batch_size, 28*28))
        xent_loss = 28*28*binary_crossentropy(x, decoded)
        kl_loss = -0.5 * K.sum(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
        return (xent_loss + kl_loss)/2/28/28

    return models, vae_loss

models, vae_loss = create_vae()
vae = models["vae"]


from keras.optimizers import Adam
vae.compile(optimizer=Adam(start_lr), loss=vae_loss)



digit_size = 28

n = 15 # Количество изображений на одной стороне рисуемого множества

from scipy.stats import norm
# Сетка узлов на скрытом пространстве
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

epochs_for_drawing = list(range(10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list([999])

def draw_manifold(generator, epoch):
    figure = np.zeros((digit_size * n, digit_size * n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.zeros((1, Z_dimension))
            z_sample[:, :2] = np.array([[xi, yi]])

            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].squeeze()
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    figure = 1-figure

    plt.figure(figsize=(15, 15))
    plt.imshow(figure, cmap='Greys_r')
    plt.grid(None)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    filename = path_to_img + 'manifold-' + str(epoch) + '.png'
    plt.savefig(filename)
    plt.close()

def draw_latent_distribution(zs, epoch, c):
    im = plt.scatter(zs[:, 0], zs[:, 1], c=c, cmap=cm.coolwarm)
    filename = path_to_img + 'ld-' + str(epoch) + '.png'
    plt.savefig(filename)
    plt.close()

latent_distrs = []
epochs = []

# Эпохи, которые будем сохранять в файл
save_epochs = list(range(100)) + list(range(100, 300, 2)) + list(range(300, 700, 5)) + list(
    range(700, 1000, 10)) + list([999])

# Модели
generator   = models["decoder"]
encoderMu   = models["z_mu"]

# Функция, которую будем запускать после каждой эпохи
def on_epoch_end(epoch, logs):
    if epoch in save_epochs:
        text_to_file = '\ni: ' + str(epoch) + ',  loss: ' +  str(logs['loss'])
        text_file.write(text_to_file)
        text_file.flush()

        latent_space = encoderMu.predict(x_test, batch_size)

        # Сохранение многообразия и распределения z для последующего сохранения в файл
        epochs.append(epoch)
        latent_distrs.append(latent_space)

        if epoch in epochs_for_drawing:
            draw_manifold(generator, epoch)
            draw_latent_distribution(latent_space, epoch, y_test)

pltfig = LambdaCallback(on_epoch_end=on_epoch_end)
tb     = TensorBoard(log_dir='./logs')

with open('data.pkl', 'wb') as f:
    pickle.dump([epochs, latent_distrs], f)

""" ======= ОБУЧЕНИЕ ======= """
vae.fit(x_train, x_train, shuffle=True, epochs=100,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[pltfig, tb],
        verbose=1)

""" == СОХРАНЕНИЕ МОДЕЛЕЙ == """
vae.save('vae.h5')
encoderMu.save('latent.h5')
generator.save('decoder.h5')
