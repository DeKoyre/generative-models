from keras import backend as K
from keras.models import Model, load_model
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

digit_size = 28
x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), digit_size, digit_size, 1))
x_test  = np.reshape(x_test,  (len(x_test),  digit_size, digit_size, 1))

y_train_cat = to_categorical(y_train).astype(np.float32)
y_test_cat  = to_categorical(y_test).astype(np.float32)
num_classes = y_test_cat.shape[1]

def plot_digits(*args, invert_colors=False):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    figure = np.zeros((digit_size * len(args), digit_size * n))

    for i in range(n):
        for j in range(len(args)):
            figure[j * digit_size: (j + 1) * digit_size,
                   i * digit_size: (i + 1) * digit_size] = args[j][i].squeeze()

    if invert_colors:
        figure = 1 - figure

    plt.figure(figsize=(2*n, 2*len(args)))
    plt.imshow(figure, cmap='Greys_r')
    plt.grid(False)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(str(target_label) + "test.png")
    plt.close()

def style_transfer(model, X, input_label, result_label):
    rows = X.shape[0]
    if isinstance(input_label, int):
        label_idx = input_label
        input_label = np.zeros((rows, 10))
        input_label[:, label_idx] = 1
    if isinstance(result_label, int):
        label_idx = result_label
        result_label = np.zeros((rows, 10))
        result_label[:, label_idx] = 1
    return model.predict([X, input_label, result_label])


n = 15
target_label = 4
generated = []
prot = x_train[y_train == target_label][:n]

transformer = model = load_model('transformer.h5')

for i in range(num_classes):
    generated.append(style_transfer(transformer, prot, target_label, i))

generated[target_label] = prot
plot_digits(*generated, invert_colors=True)
