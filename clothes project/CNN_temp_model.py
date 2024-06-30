# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 15:20:05 2023

@author: safak
"""
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import balanced_accuracy_score
import sklearn
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import pandas as pd
from keras.optimizers import SGD
import matplotlib.pyplot as plt

#verileri elde etme

IMG_WIDTH = 90
IMG_HEIGHT = 113

catg_name=["ayakkabi","ceket","esofman","gomlek","kazak","pantolon","sort","sweatshirt","tshirt"]
gender_name="erkek"

path="C:\\Users\\safak\\spyder projects\\clothes\\DATASETS\\"+gender_name

images=[]
like=[]

def flatten_extend(matrix):
     flat_list = []
     for row in matrix:
         flat_list.extend(row)
     return flat_list
 

"""for i in catg_name:
    path_2="C:\\Users\\safak\\spyder projects\\clothes\\DATASETS\\"+gender_name+"\\DATA_"+gender_name+"_"+i
    l=os.listdir(path_2)
    df=pd.read_csv(path_2+"\\"+l[1])    
    imgs=os.listdir(path_2+"\\"+l[0])
    like.append(df[df.columns[3]].values)    
    for j in imgs:
        img=cv2.imread(path_2+"\\"+l[0]+"\\"+j)
        img=cv2.resize(img,(32,32))
        images.append(img)"""
        
        
path_2="C:\\Users\\safak\\spyder projects\\clothes\\DATASETS\\"+gender_name+"\\DATA_"+gender_name+"_tshirt"
l=os.listdir(path_2)
df=pd.read_csv(path_2+"\\"+l[1])    
imgs=os.listdir(path_2+"\\"+l[0])
like.append(df[df.columns[3]].values)    
for j in imgs:
    img=cv2.imread(path_2+"\\"+l[0]+"\\"+j)
    img=cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
    images.append(img)
        
likes=flatten_extend(like)
        
images=np.array(images)
likes=np.array(likes)
        
#verileri ayırma
x_train2,x_test2,y_train2,y_test2=train_test_split(images,likes,test_size=0.5,random_state=42)
x_train2,x_valid2,y_train2,y_valid2=train_test_split(x_train2,y_train2,test_size=0.3,random_state=42)

#visualization
"""fig,axes=plt.subplots(3,1,figsize=(7,7))
fig.subplots_adjust(hspace=0.5)
sns.countplot(y_train,ax=axes[0])
axes[0].set_title("y_train")
sns.countplot(y_test,ax=axes[1])
axes[1].set_title("y_test")
sns.countplot(y_valid,ax=axes[2])
axes[2].set_title("y_valid")"""

"""def preProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img

x_train=np.array(list(map(preProcess,x_train)))
x_test=np.array(list(map(preProcess,x_test)))
x_valid=np.array(list(map(preProcess,x_valid)))

x_train=x_train.reshape(-1,32,32,1)
x_test=x_test.reshape(-1,32,32,1)
x_valid=x_valid.reshape(-1,32,32,1)"""

x_train2=x_train2.reshape(-1,32,32)
x_test2=x_test2.reshape(-1,32,32)
x_valid2=x_valid2.reshape(-1,32,32)

def preProcess(img):
    scaler=RobustScaler()
    scaler.fit(img)
    img=scaler.transform(img)
    return img

x_train2=np.array(list(map(preProcess,x_train2)))
x_test2=np.array(list(map(preProcess,x_test2)))
x_valid2=np.array(list(map(preProcess,x_valid2)))

x_train=x_train.reshape(-1,32,32,3)
x_test=x_test.reshape(-1,32,32,3)
x_valid=x_valid.reshape(-1,32,32,3)

#data generate
data_gen=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,
                            zoom_range=0.1,rotation_range=10)
data_gen.fit(x_train)

#model inşası
"""model=Sequential()
model.add(Conv2D(input_shape=(32,32,3),filters=8,kernel_size=(5,5),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16,kernel_size=(3,3), activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=256,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation="linear"))

model.compile(loss="mean_squared_error",optimizer=("Adam"),metrics=["accuracy"])
batch_size=45

hist=model.fit(data_gen.flow(x_train,y_train,batch_size=batch_size),validation_data=(x_valid,y_valid),epochs=15,steps_per_epoch=1,shuffle=1)"""
BATCH_SIZE = 16

IMG_CHANNELS = 3

input_img = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Tabular data
input_tab = tf.keras.layers.Input(shape=(12,))
y = tf.keras.layers.Dense(16, activation='relu')(input_tab)
y = tf.keras.layers.Dense(32, activation='relu')(y)

# Concatenate models outputs
concatenated = tf.keras.layers.concatenate([x, y], axis=-1)

output_score = tf.keras.layers.Dense(1, activation=None)(concatenated)

# Build general model with Keras Functional API
model = tf.keras.models.Model([input_img, input_tab], output_score)
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

print(model.summary())
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=False)

hist=model.fit(data_gen.flow(x_train2,y_train2,batch_size=BATCH_SIZE),validation_data=(x_valid2,y_valid2),epochs=15,steps_per_epoch=1,shuffle=1)

#-------------------------------*******************************-------------------
"""pickle_out=open("trained_model_misir.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()"""

#model.save("misir_cnn_3")

#model=tf.keras.models.load_model('misir_cnn_5')

#değerlendirme
plt.figure()
plt.plot(hist.history["loss"],label="eğitim loss")
plt.plot(hist.history["val_loss"],label="validation loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"],label="eğitim accuracy")
plt.plot(hist.history["val_accuracy"],label="validation accuracy")
plt.legend()
plt.show()

score=model.evaluate(x_test,y_test,verbose=1)
print("test loss: %{}".format(score[0]))
print("test accuracy: &{}".format(score[1]))

y_pred=model.predict(x_test)
y_pred_class=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_test,axis=1)
cm=confusion_matrix(y_true,y_pred_class)

f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("confusion matrix (cm)")
plt.show()


y_pred=model.predict(x_train)
y_pred_class=np.argmax(y_pred,axis=1
                       )
y_true=np.argmax(y_train,axis=1)
cm=confusion_matrix(y_true,y_pred_class)

f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("confusion matrix (cm)")
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()






































