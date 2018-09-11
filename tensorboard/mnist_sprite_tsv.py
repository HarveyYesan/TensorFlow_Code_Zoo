import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = "/home/harvey/tftest"
SPRITE_FILE = 'mnist_sprite.jpg'
META_FILE = 'mnist_meta.tsv'

# generate sprite file
def create_sprite_file(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    # edge length of sprite images
    m = int(np.ceil(np.sqrt(images.shape[0])))
    # init sprite images
    sprite_image = np.ones((img_h*m, img_w*m))

    for i in range(m):
        for j in range(m):
            cur = i * m + j
            if cur < images.shape[0]:
                sprite_image[i*img_h:(i+1)*img_h, j*img_w:(j+1)*img_w] = images[cur]
    return sprite_image

# generate sprite image
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
to_visualise = 1 - np.reshape(mnist.test.images, (-1, 28, 28))
sprite_image = create_sprite_file(to_visualise)

path_for_sprite = os.path.join(LOG_DIR, SPRITE_FILE)
plt.imsave(path_for_sprite, sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray')

# generate tsv file
path_for_tsv = os.path.join(LOG_DIR, META_FILE)
with open(path_for_tsv, 'w') as f:
    f.write('Index\tLabel\n')
    for index, label in enumerate(mnist.test.labels):
        f.write('%d\t%d\n' % (index, label))





