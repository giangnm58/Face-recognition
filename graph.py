import numpy as np
from keras.models import load_model, Model, Sequential, Input
import matplotlib.pyplot as plt
from keras import backend as K
import re
from sklearn import manifold
from sklearn.decomposition import PCA 

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


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def triplet_loss(y_true, y_pred, alpha = 0.3):
    anchor, positive, negative = y_pred[:,0:40], y_pred[:,40:80], y_pred[:,80:120]
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    basic_loss = pos_dist-neg_dist+alpha 
    loss = K.maximum(basic_loss,0.0)
    return loss

size = 2
total_sample_size = 200
catagory = 40
def get_data(size, total_sample_size):
    image = read_image('/home/wangyunhao/Desktop/att-database-of-faces/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    image = image[::size, ::size]
    dim1 = image.shape[0] 
    dim2 = image.shape[1] 
    x_geuine_pair = np.zeros([total_sample_size, 1, dim1, dim2])
    count=0 
    for i in range(40): 
        for j in range(5): 
            img1 = read_image('/home/wangyunhao/Desktop/att-database-of-faces/s' + str(i+1) + '/' + str(j+1) + '.pgm', 'rw+')
            img1 = img1[::size,::size]
            x_geuine_pair[count,0,:,:] = img1
            count +=1
    X = x_geuine_pair.astype('float32')/255
    return X

data = get_data(size,total_sample_size)
model_cont = load_model('recognition_contrastive.h5',custom_objects={'contrastive_loss': contrastive_loss})
model_trip = load_model('recognition_triplet.h5',custom_objects={'triplet_loss': triplet_loss})

input_layer = Input(shape=(1,56,46))
base = model_cont.get_layer('sequential_1')(input_layer)
model_c = Model(input_layer,base)



input_layer_t = Input(shape=(1,56,46))
base_t = model_trip.get_layer('sequential_1')(input_layer_t)
model_t = Model(input_layer_t,base_t)


c_predict = model_c.predict(data)
t_predict = model_t.predict(data)



label = np.array([n//5 for n in range(200)])



C_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(c_predict)

#Data Visualization
c_min, c_max = C_tsne.min(0), C_tsne.max(0)
C_norm = (C_tsne - c_min) / (c_max - c_min)  #Normalize
plt.figure(figsize=(10, 10))
for i in range(C_norm.shape[0]):
    plt.text(C_norm[i, 0], C_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.title("Siamese Network using contrastive loss")
plt.show()


T_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(t_predict)

t_min, t_max = T_tsne.min(0), T_tsne.max(0)
T_norm = (T_tsne - t_min) / (t_max - t_min)  #Normalize
plt.figure(figsize=(10, 10))
for i in range(T_norm.shape[0]):
    plt.text(T_norm[i, 0], T_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.title("Siamese Network using triplet loss")        
plt.show()


pca=PCA(n_components=2)
C_PCA =pca.fit_transform(c_predict)
c_min, c_max = C_PCA.min(0), C_PCA.max(0) 
C_norm = (C_PCA - c_min) / (c_max - c_min)  #Normalize 
plt.figure(figsize=(10, 10)) 
for i in range(C_norm.shape[0]): 
    plt.text(C_norm[i, 0], C_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]),  
                fontdict={'weight': 'bold', 'size': 9}) 
plt.xticks([]) 
plt.yticks([])
plt.title("Siamese Network using contrastive loss")  
plt.show() 


T_PCA =pca.fit_transform(t_predict)
t_min, t_max = T_PCA.min(0), T_PCA.max(0) 
T_norm = (T_PCA - t_min) / (t_max - t_min)  #Normalize 
plt.figure(figsize=(10, 10)) 
for i in range(T_norm.shape[0]): 
    plt.text(T_norm[i, 0], T_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]),  
                fontdict={'weight': 'bold', 'size': 9}) 
plt.xticks([]) 
plt.yticks([])
plt.title("Siamese Network using triplet loss")  
plt.show() 
