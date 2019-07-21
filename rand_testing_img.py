import scipy.io as sio
import random
import numpy as np
import sys
from PIL import Image


def load_dataset(file_path):
    dataset = sio.loadmat(file_path)
    images = dataset['X']
    return images


def get_random_img(images):
    rand_img = []
    for i in range(0, 6):
        rdm = random.randrange(0, images.shape[3])
        image = images[:, :, :, rdm]
        img = Image.fromarray(image, 'RGB')
        image_name = str(rdm) + '.png'
        image_path = './static/' + image_name
        img.save(image_path)
        rand_img.append((image_name, image, rdm))
    return np.asarray(rand_img)


def main(dataset_path):
    images = load_dataset(dataset_path)
    rand_images = get_random_img(images)
    # print(test_model.main('../HelloTFCPU/models/res_net_test_SVHN_with_all_goodies_smaller_32_20_grayscale_extra', 'dataset/train_32x32', 201))
    return rand_images


if __name__ == "__main__":
    main(sys.argv[1])
