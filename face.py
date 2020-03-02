import cv2
from PIL import Image
import os 




face_cascade = cv2.CascadeClassifier('/home/wangyunhao/Desktop/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
base = '/home/wangyunhao/Desktop/s42'
num = 1
for name in os.listdir(base):
    path =os.path.join(base,name)
    img = cv2.imread(path,0)
    faces = face_cascade.detectMultiScale(img,1.4,5)
    for (x,y,w,h) in faces:
        filename = str(num) +".pgm"
        image = Image.open(path).convert('L')
        imageCrop = image.crop((x,y,x+w,y+h))
        imageResize = imageCrop.resize((92,112),Image.ANTIALIAS)
        imageResize.save(filename)
    num+=1


