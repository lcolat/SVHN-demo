import numpy as np
import scipy.io as sio
from tensorflow.python.keras.models import load_model


def load_image(file_path, image_index):
    dataset = sio.loadmat(file_path)
    images = dataset['X']
    image = images[:, :, :, image_index]
    return image


def get_expected_result(dataset_path, image_index):
    dataset = sio.loadmat(dataset_path)
    y_test = dataset['y']
    return y_test[image_index]


def rgb2gray(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])


def main(model_path, dataset_path, image_index):
    image = load_image(dataset_path, image_index)
    model = load_model(model_path)
    expected_result = get_expected_result(dataset_path, image_index)
    image = rgb2gray(image)
    image = np.reshape(image, (-1, 1024))
    predict_val = model.predict(image)
    return str(expected_result[0]), str(np.argmax(predict_val)), str(round(np.amax(predict_val) * 100, 2)) + '%'


if __name__ == "__main__":
    # 72012train28176|60284
    # image_ind = 72012
    # train_data = sio.loadmat('train_32x32.mat')
    # model = load_model('../HelloTFCPU/models/res_net_test_SVHN_with_all_goodies_smaller_32_20_grayscale_extra')
    # model = load_model('../HelloTFCPU/models/conv2d_SVHN')
    # img = Image.fromarray(x_train, 'RGB')
    # img.save('my.png')
    # plt.imshow(x_train)
    # plt.show()
    # x_train = rgb2gray(x_train)
    # img = Image.fromarray(x_train)
    # img.save('myGray.png')
    # plt.imshow(x_train)
    # plt.show()
    main()
