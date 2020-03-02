import re
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
import tensorflow as tf

def read_image(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    header, width, height, maxval = re.search(
        b"(^P5\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    return np.frombuffer(buffer,dtype='u1' if int(maxval) < 256 else byteorder+'u2',
        count=int(width)*int(height),
        offset=len(header)
        ).reshape((int(height), int(width)))

size = 2
total_sample_size = 12000

def get_data(size, total_sample_size):
    image = read_image('/home/wangyunhao/Desktop/att-database-of-faces/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    image = image[::size, ::size]
    dim1 = image.shape[0] 
    dim2 = image.shape[1] 
    count = 0
    x_geuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2]) 
    y_genuine = np.zeros([total_sample_size, 1]) 
    for i in range(40): 
        for j in range(int(total_sample_size/40)): 
            ind1 = 0
            ind2 = 0
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)
            img1 = read_image('/home/wangyunhao/Desktop/att-database-of-faces/s' + str(i+1) + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            img2 = read_image('/home/wangyunhao/Desktop/att-database-of-faces/s' + str(i+1) + '/' + str(ind2 + 1) + '.pgm', 'rw+')
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]
            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2
            y_genuine[count] = 1
            count += 1
    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])
    for i in range(int(total_sample_size/10)): 
        for j in range(10):
            while True:
                ind1 = np.random.randint(40)
                ind2 = np.random.randint(40)
                if ind1 != ind2:
                    break
            img1 = read_image('/home/wangyunhao/Desktop/att-database-of-faces/s' + str(ind1+1) + '/' + str(j + 1) + '.pgm', 'rw+')
            img2 = read_image('/home/wangyunhao/Desktop/att-database-of-faces/s' + str(ind2+1) + '/' + str(j + 1) + '.pgm', 'rw+')
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]
            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            y_imposite[count] = 0
            count += 1
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0).astype('float32')/255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)
    return X, Y

X, Y = get_data(size, total_sample_size)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25) 

def build_base_network(input_shape):
    seq = Sequential()
    
    seq.add(Convolution2D(6, (3,3), input_shape=input_shape,border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2))) 
    seq.add(Dropout(.25))
    
    seq.add(Convolution2D(12, (3,3), border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th')) 
    seq.add(Dropout(.25))

    seq.add(Flatten())
    seq.add(Dense(512, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(40, activation='relu'))
    return seq

input_dim = x_train.shape[2:] 
img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

base_network = build_base_network(input_dim)
feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

#distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
distance = Lambda(euclidean_distance)([feat_vecs_a, feat_vecs_b])

epochs = 13
rms = RMSprop()
adam = Adam()

model = Model(input=[img_a, img_b], output=distance)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    
model.compile(loss=contrastive_loss, optimizer=adam)

img_1 = x_train[:, 0]
img_2 = x_train[:, 1] 

model.fit([img_1, img_2], y_train, validation_split=.25, batch_size=128, verbose=1, nb_epoch=epochs)

model.save('recognition_contrastive.h5')

pred = model.predict([x_test[:, 0], x_test[:, 1]])

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()

print(compute_accuracy(pred, y_test))