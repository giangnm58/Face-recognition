import re
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, concatenate
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
    x_geuine_pair = np.zeros([total_sample_size, 3, 1, dim1, dim2]) 
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
            while True:
                ind3 = np.random.randint(40)
                if ind3 !=i:
                    break
            ind4 = np.random.randint(10)
            img3 = read_image('/home/wangyunhao/Desktop/att-database-of-faces/s' + str(ind3+1) + '/' + str(ind4+ 1) + '.pgm', 'rw+')

            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]
            img3 = img3[::size, ::size]

            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2
            x_geuine_pair[count, 2, 0, :, :] = img3
            count += 1
    X = x_geuine_pair.astype('float32')/255
    Y = y_genuine
    return X, Y
X, Y = get_data(size, total_sample_size)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25) 

def triplet_loss(y_true, y_pred, alpha = 0.3):
    anchor, positive, negative = y_pred[:,0:40], y_pred[:,40:80], y_pred[:,80:120]
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss

def build_base_network(input_shape):
    seq = Sequential()
    
    #convolutional layer 1
    seq.add(Convolution2D(6, (3,3), input_shape=input_shape,border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2))) 
    seq.add(Dropout(.25))
    
    #convolutional layer 2
    seq.add(Convolution2D(12, (3,3), border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th')) 
    seq.add(Dropout(.25))

    #flatten 
    seq.add(Flatten())
    seq.add(Dense(512, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(40, activation='relu'))
    return seq

input_dim = x_train.shape[2:] #(1,56,46)
img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)
img_c = Input(shape=input_dim)

base_network = build_base_network(input_dim)
feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)
feat_vecs_c = base_network(img_c)


merged_vector = concatenate([feat_vecs_a, feat_vecs_b, feat_vecs_c], axis=-1, name='merged_layer')

epochs = 13
rms = RMSprop()
adam = Adam()

model = Model(input=[img_a, img_b,img_c], output=merged_vector)
model.compile(loss=triplet_loss, optimizer=adam)

img_1 = x_train[:, 0]
img_2 = x_train[:, 1] 
img_3 = x_train[:, 2]

model.fit([img_1, img_2, img_3], y_train, validation_split=.25, batch_size=128, verbose=1, nb_epoch=epochs)

model.save('recognition_triplet.h5')

pred = model.predict([x_test[:, 0], x_test[:, 1], x_test[:,2]])

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def compute_accuracy(predictions, test_len):
    sess = tf.InteractiveSession()
    x = euclidean_distance((predictions[:,0:40],predictions[:,40:80])).eval()
    recall = np.sum(np.where(x<0.5,1,0))/ test_len

    y = euclidean_distance((predictions[:,0:40],predictions[:,80:120])).eval()
    precision = np.sum(np.where(y>0.5,1,0))/ test_len
    return recall,precision

r,p = compute_accuracy(pred, len(x_test))

print(r,p)
