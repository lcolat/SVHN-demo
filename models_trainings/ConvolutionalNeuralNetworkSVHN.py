from tensorflow.python.keras.activations import *
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.datasets import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.losses import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.utils import *
import scipy.io as sio


def create_conv_net():
    model = Sequential()
    model.add(Reshape((32, 32, 1), input_shape=(1024,)))
    model.add(Conv2D(16, 3, padding='same', activation=relu))
    model.add(AveragePooling2D())
    model.add(Conv2D(32, 3, padding='same', activation=relu))
    model.add(AveragePooling2D())
    model.add(Conv2D(64, 3, padding='same', activation=relu))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(10, activation=softmax))
    model.compile(loss=categorical_crossentropy, optimizer=adam(), metrics=['accuracy'])
    return model


def rgb2gray(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])


def reformat_images(images):
    images_qty = images.shape[3]
    new_images = []
    for i in range(0, images_qty):
        image = rgb2gray(images[:,:,:,i])
        new_images.append(image)
        print(str(i) + '/' + str(images_qty))
    return np.asarray(new_images)


def load_data(train_file, test_file):
    train_data = sio.loadmat(train_file)
    test_data = sio.loadmat(test_file)
    x_train = train_data['X']
    x_test = test_data['X']
    y_train = train_data['y'] % 10
    y_test = test_data['y'] % 10
    print(x_train.shape)
    x_train = reformat_images(x_train)
    print(x_train.shape)
    x_test = reformat_images(x_test)
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data('extra_32x32.mat', 'test_32x32.mat')
    x_train = np.reshape(x_train, (-1, 1024))
    x_test = np.reshape(x_test, (-1, 1024))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    model = create_conv_net()

    model_name = 'conv2d_SVHN_extra'

    tb_callback = TensorBoard('./logs/' + model_name)

    print(model.summary())

    model.fit(x_train, y_train, epochs=100, batch_size=4096,
              callbacks=[tb_callback], validation_data=(x_test, y_test))

    plot_model(model, './models/' + model_name + '.png',
               show_shapes=True,
               show_layer_names=True)
    model.save('./models/' + model_name)
