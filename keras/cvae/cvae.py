from keras.callbacks import LambdaCallback, ReduceLROnPlateau, TensorBoard
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import concatenate, BatchNormalization, Dense, Dropout, Flatten, Input, Lambda, Reshape
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.utils import to_categorical
from keras import backend as K

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

path_to_img = "images/"
path_to_manifolds = "images/manifolds/"
path_to_distributions = "images/distributions/"

def makeDir(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

makeDir(path_to_img)
makeDir(path_to_manifolds)
makeDir(path_to_distributions)

text_file = open("cvae_output.txt", "w")
text_to_file = ''

(x_train, y_train), (x_test, y_test) = mnist.load_data()
im_size = 28

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), im_size, im_size, 1))
x_test  = np.reshape(x_test,  (len(x_test),  im_size, im_size, 1))

y_train_cat = to_categorical(y_train).astype(np.float32)
y_test_cat  = to_categorical(y_test).astype(np.float32)
num_classes = y_test_cat.shape[1]

for i in range(num_classes):
    path_m = path_to_manifolds + str(i) + "/"
    path_d = path_to_distributions + str(i) + "/"
    makeDir(path_m)
    makeDir(path_d)

batch_size = 500
Z_dimension = 8
dropout_rate = 0.3
start_learning_rate = 0.001

def create_cvae():
    models = {}

    """ ======= ЭНКОДЕР ======= """
    input_img = Input(shape=(im_size, im_size, 1))
    flatten_img = Flatten()(input_img)
    input_label = Input(shape=(num_classes,), dtype='float32')

    x = concatenate([flatten_img, input_label])
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    z_mu = Dense(Z_dimension)(x)
    z_sigma = Dense(Z_dimension)(x)

    def sampling(args):
        z_mu, z_sigma = args
        epsilon = K.random_normal(shape=(batch_size, Z_dimension), mean=0., stddev=1.0)
        return z_mu + K.exp(z_sigma / 2) * epsilon
    l = Lambda(sampling, output_shape=(Z_dimension,))([z_mu, z_sigma])

    models["encoder"]  = Model([input_img, input_label], l, 'Encoder')
    models["z_mu"] = Model([input_img, input_label], z_mu, 'Enc_z_mu')
    models["z_sigma"] = Model([input_img, input_label], z_sigma, 'Enc_z_sigma')

    """ ======= ДЕКОДЕР ======= """
    z = Input(shape=(Z_dimension, ))
    input_label_d = Input(shape=(num_classes,), dtype='float32')
    x = concatenate([z, input_label_d])

    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(im_size*im_size, activation='sigmoid')(x)
    decoded = Reshape((im_size, im_size, 1))(x)

    models["decoder"] = Model([z, input_label_d], decoded, name='Decoder')
    models["cvae"]    = Model([input_img, input_label, input_label_d],
                              models["decoder"]([models["encoder"]([input_img, input_label]), input_label_d]),
                              name="CVAE")
    models["style_transformer"] = Model([input_img, input_label, input_label_d],
                               models["decoder"]([models["z_mu"]([input_img, input_label]), input_label_d]),
                               name="style_transformer")

    def vae_loss(x, decoded):
        x = K.reshape(x, shape=(batch_size, im_size*im_size))
        decoded = K.reshape(decoded, shape=(batch_size, im_size*im_size))
        xent_loss = im_size*im_size*binary_crossentropy(x, decoded)
        kl_loss = -0.5 * K.sum(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
        return (xent_loss + kl_loss)/2/im_size/im_size

    return models, vae_loss

models, vae_loss = create_cvae()
cvae = models["cvae"]






from keras.optimizers import Adam
cvae.compile(optimizer=Adam(start_learning_rate), loss=vae_loss)



digit_size = im_size

n = 15 # Количество изображений на одной стороне рисуемого множества

from scipy.stats import norm
# Сетка узлов на скрытом пространстве
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

epochs_for_drawing = list(range(10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list([999])

def draw_manifold(generator, label, epoch):
    figure = np.zeros((digit_size * n, digit_size * n))

    input_label = np.zeros((1, 10))
    input_label[0, label] = 1

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.zeros((1, Z_dimension))
            z_sample[:, :2] = np.array([[xi, yi]])

            x_decoded = generator.predict([z_sample, input_label])
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
    filename = path_to_manifolds + str(label) + '/' + str(label) + '-manifold-' + str(epoch) + '.png'
    plt.savefig(filename)
    plt.close()

def draw_latent_distribution(z, epoch, label):
        im = plt.scatter(z[:, 0], z[:, 1])
        im.axes.set_xlim(-5, 5)
        im.axes.set_ylim(-5, 5)
        filename = path_to_distributions + str(label) + '/' + str(label) + '-ld-' + str(epoch) + '.png'
        plt.savefig(filename)
        plt.close()


# Эпохи, которые будем сохранять в файл
save_epochs = list(range(100)) + list(range(100, 300, 2)) + list(range(300, 700, 5)) + list(
    range(700, 1000, 10)) + list([999])

# Модели
generator   = models["decoder"]
encoderMu   = models["z_mu"]
transformer = models["style_transformer"]

# Функция, которую будем запускать после каждой эпохи
def on_epoch_end(epoch, logs):
    if epoch in save_epochs:
        # Сохранение в файл значений функции потерь
        text_to_file = '\ni: ' + str(epoch) + ',  loss: ' +  str(logs['loss'])
        text_file.write(text_to_file)
        text_file.flush()

        # Рисование многообразий
        if epoch in epochs_for_drawing:
            for label in range(num_classes):
                draw_manifold(generator, label, epoch)

                idxs = y_test == label
                z_predicted = encoderMu.predict([x_test[idxs], y_test_cat[idxs]], batch_size)
                draw_latent_distribution(z_predicted, epoch, label)

pltfig = LambdaCallback(on_epoch_end=on_epoch_end)
tb     = TensorBoard(log_dir='./logs')

""" ======= ОБУЧЕНИЕ ======= """
cvae.fit([x_train, y_train_cat, y_train_cat], x_train, shuffle=True, epochs=1000,
         batch_size=batch_size,
         validation_data=([x_test, y_test_cat, y_test_cat], x_test),
         callbacks=[pltfig, tb],
         verbose=1)


""" == СОХРАНЕНИЕ МОДЕЛЕЙ == """
cvae.save('cvae.h5')
encoderMu.save('latent.h5')
generator.save('decoder.h5')
transformer.save('transformer.h5')
