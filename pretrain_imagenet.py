import keras
import seaborn as sns
from tensorflow.keras.models import Sequential
import math
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Layer, Add
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
import cv2
import plotly.graph_objects as go
import os
import mlflow
import mlflow.tensorflow
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.applications import mobilenet, inception_v3
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from time import time

df = pd.read_csv('../../cleaneddataset2.csv') #Reading data

mlflow.tensorflow.autolog() #Logging modeld parameters and metrices

df = df.drop(['ZosteraMarina', 'CodiumFragile', 'Fucusspp', 'Turf', 'ChondusCrispus',
       'PalmariaPalmata', 'CorallineAlgae', 'Desmarestiaspp',
       'UnifentifiedAlgae', 'Bedrock', 'Boulder', 'Cobble', 'Pebble', 'Sand',
       'PlumbLine', 'Unknown', 'MixedChondrusTurf',
       'CorallinaOfficinalis', 'MixedChondrusTurfCorallina', 'MixedChondrusCoralina',
       'MixedTurfCorallina', 'TotalPts', 'Kelp'], axis=1) #dropping individual species

df['SaccharinaLatissima']=df.iloc[:,[12,19]].sum(axis=1) # Summing two species of SL
df['OtherKelp']=df.iloc[:,[15,16,17,18]].sum(axis=1) #Summing 4 species into OtherKelp

df = df.drop(['AlariaEsculenta', 'SacchorizaDermatodea', 'UnidentifiedKelp',
       'MembraniporaMembranacea', 'JuvenileSaccharinaLatissima'], axis=1) # Dropping previously summed individual species

df.iloc[:, 12:] = df.iloc[:, 12:].fillna(0) #Filling NA values with 0 in dataframe
df.iloc[:, 12:] = df.iloc[:, 12:].div(df.iloc[:, 12:].sum(axis=1), axis=0) #Transforming each rows to sum equals to 1

def variance_of_laplacian(image): #Function to return degree of blur for an image
	return cv2.Laplacian(image, cv2.CV_64F).var()

train_image = []
train_labels = []
test_image = []
test_labels = []
scale_percent = 30 # Scale image to 30% of original size
# dim = (811, 456)
IN_SHAPE = (456, 811, 3)

#Another way is to loop over every folder and create dataframe for that folder then loop over that dataframe
for i in tqdm(df.itertuples()): #Iterating over each row
    label = list(i[13:]) #Labels for training

    #Converting values so that it matches the names in thee folder of the images and name of the images
    island = i.SiteName.replace(" ", "")
    m = str(i[2]) + 'm'
    diver = m + '_' + i.Diver
    im_num = str(i.ImageNumber).zfill(3) #Analyzed Image Number
    im_name = island + str(i[2]) + i.Diver[0] + '_' + im_num #Image Name

    path = "../../videos/" + island + '/' + diver + '/'
    directory = os.listdir(path)


    #Taking images from analyzed Barren Island folder as we dont have video of Barren island for neareset frames
    if(island=='BarrenI' and diver=='6m_KA'):
        img3 = cv2.imread(path + str(i.ImageNumber) + '.jpg') # Reading image
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB) #Converting it into RGB format
        #Resizing it into 30% scale of original image
        width3 = int(img3.shape[1] * scale_percent / 100)
        height3 = int(img3.shape[0] * scale_percent / 100)
        dim3 = (width3, height3)
        resized3 = cv2.resize(img3, dim3)
        train_image.append(resized3)
        train_labels.append(label)

    #Images of The Moll for testing purpose
    elif(island=='TheMoll'):
        images = []
        #Appending each image of "The Moll" from folder
        for j in sorted(directory):
            num = j.split('.jpg')[0]
            images.append(num)

        #Every folder has frame extracted every 10 sec from the video which  has '-' in its image name so creating dictionary with key as that frame and values as its nearest frames
        dict1 = {}
        for k,l in enumerate(images): # Iterating over every image
            if('-' in l):
                first = int(l.split('-')[0]) #The number before '-' of image shows the number of the image among every other images while the number after '-' shows the number of image among frames extracted every 10 sec
                if(first==0): # If the extracted frame is the first image in the folder than take the next 10 frames
                    dict1[l] = [m for m in range(1,11)]
                else: #If not then take 10 frames before and 10 after the extracted frame
                    before = first - 10 #Image number from 10 frames before of extracted frame
                    after = first + 10 #Image number from 10 frames after of extracted frame
                    list1 = [n for n in range(before, first)] #List of images from 10 frames before to current extracted frame
                    list2 = [o for o in range(first, after)] #List of images from current extracted frame to 10 frames after
                    list3 = list1 + list2 #Adding both the list
                    dict1[l] = list3

        for p in dict1: #Looping over dictionary
            key = int(p.split('-')[1]) #Taking image number of extracted frame
            if(key==i.ImageNumber): #If the number equals to the ImageNumber(number of analyzed frame), then append it in testing list
                img1 = cv2.imread(path + p + '.jpg')
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                width = int(img1.shape[1] * scale_percent / 100)
                height = int(img1.shape[0] * scale_percent / 100)
                dim=(width,height)
                resized1 = cv2.resize(img1, dim)
                test_image.append(resized1)
                test_labels.append(label)
                for q in dict1[p]: #Looping over values of dictionary and appending it in testing list
                    img2 = cv2.imread(path + str(q) + '.jpg')
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    width2 = int(img2.shape[1] * scale_percent / 100)
                    height2 = int(img2.shape[0] * scale_percent / 100)
                    dim2 = (width2, height2)
                    resized2 = cv2.resize(img2, dim2)
                    test_image.append(resized2)
                    test_labels.append(label)

    #Same steps as for The Moll but for every island other than Moll for training purpose
    else:
        images = []
        for j in sorted(directory):
            num = j.split('.jpg')[0]
            images.append(num)

        dict1 = {}
        for k,l in enumerate(images):
            if('-' in l):
                first = int(l.split('-')[0])
                if(first==0):
                    dict1[l] = [m for m in range(1,11)]
                else:
                    before = first - 10
                    after = first + 10
                    list1 = [n for n in range(before, first)]
                    list2 = [o for o in range(first, after)]
                    list3 = list1 + list2
                    dict1[l] = list3

        for p in dict1:
            key = int(p.split('-')[1])
            if(key==i.ImageNumber):
                img1 = cv2.imread(path + p + '.jpg')
                gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                fm = variance_of_laplacian(gray)
                if(fm>30):
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                    width = int(img1.shape[1] * scale_percent / 100)
                    height = int(img1.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    resized1 = cv2.resize(img1, dim)
                    train_image.append(resized1)
                    train_labels.append(label)
                    for q in dict1[p]:
                        img2 = cv2.imread(path + str(q) + '.jpg')
                        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                        width2 = int(img2.shape[1] * scale_percent / 100)
                        height2 = int(img2.shape[0] * scale_percent / 100)
                        dim2 = (width2, height2)
                        resized2 = cv2.resize(img2, dim2)
                        train_image.append(resized2)
                        train_labels.append(label)


#Converting training and testing images into numpy array
X_train = np.array(train_image, dtype='float32')
X_test = np.array(test_image, dtype='float32')

#Converting training and testing labels into numpy array
y_train = np.array(train_labels, dtype='float32')
y_test = np.array(test_labels, dtype='float32')

#For reducing high computation, normalize the values between 0 and 1
X_train = X_train/255.0
X_test = X_test/255.0

#Pretrain Mobilenet model with imagenet weights
mobilenet_model = mobilenet.MobileNet(include_top=False, input_shape=IN_SHAPE, weights='imagenet')
x=mobilenet_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(8,activation='softmax')(x)

model=Model(inputs=mobilenet_model.input,outputs=preds)

#Using last 20 layers for training
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

#Stochastic Gradient Descent optimizer
sgd = optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd,loss=tf.keras.losses.KLDivergence(), metrics=['accuracy',tf.keras.metrics.AUC()]) #Compiling model with sgd optimizer and KLDivergence loss

early_stopping = EarlyStopping(patience=15, monitor='val_loss', verbose=1) #To stop training model when loss does not decreasing after 15 epochs
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1) #Reduce learning rate if model doesnot improve  after 10 epochs
checkpoint = ModelCheckpoint("moll_pretrain_mobilenet.ckpt", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
#Save model when the loss improves from previous one

#Parameters for ImageDataGenerator for augmentation
datagen_args = dict(
        brightness_range=[0.2,1.0],
        rotation_range=30,
        shear_range = 0.2,
        horizontal_flip=True,
        vertical_flip =True,
        preprocessing_function=preprocess_input)

image_datagen = ImageDataGenerator(**datagen_args)

train_generator = image_datagen.flow(X_train, y_train, batch_size=32)
history = model.fit_generator(train_generator,
        validation_data=(X_test, y_test),
        epochs=50,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        workers=4)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=list(range(len(history.history['loss']))), y=history.history['loss'], name='train loss', mode='lines'))
fig2.add_trace(go.Scatter(x=list(range(len(history.history['val_loss']))), y=history.history['val_loss'], name='test loss', mode='lines'))
fig2.update_layout(xaxis_title='Epochs', yaxis_title='loss')
go.Figure.write_html(fig2,"loss_pretrain_mobilenet.html")

proba = model.predict(X_test)
mape = mean_absolute_error(y_test, proba)
print(mape)

#Absolute error between predicted labels and actual labels
sl = []
ld=[]
ac=[]
algae1 = []
algae2 =[]
rock = []
unknown = []
ok = []
avrg = []
for i, j in zip(proba, y_test):
    sl_mae = np.round(abs(i[0]-j[0]), 3)
    ld_mae = np.round(abs(i[1]-j[1]), 3)
    ac_mae = np.round(abs(i[2]-j[2]), 3)
    a1_mae = np.round(abs(i[3]-j[3]), 3)
    a2_mae = np.round(abs(i[4]-j[4]), 3)
    r_mae = np.round(abs(i[5]-j[5]), 3)
    u_mae = np.round(abs(i[6]-j[6]), 3)
    ok_mae = np.round(abs(i[7]-j[7]), 3)
    avg = (sl_mae + ld_mae + ac_mae + a1_mae + a2_mae + r_mae + u_mae)/7

    sl.append(sl_mae)
    ld.append(ld_mae)
    ac.append(ac_mae)
    algae1.append(a1_mae)
    algae2.append(a2_mae)
    rock.append(r_mae)
    unknown.append(u_mae)
    ok.append(ok_mae)
    avrg.append(avg)

k1 = sum(sl)/len(sl)
k2 = sum(ld)/len(ld)
k3 = sum(ac)/len(ac)
a1 = sum(algae1)/len(algae1)
a2 = sum(algae2)/len(algae2)
r = sum(rock)/len(rock)
u = sum(unknown)/len(unknown)
otherk = sum(ok)/len(ok)

print('AVERAGE Errors:  ',k1,k2,k3,a1,a2,r,u,otherk)
fig3 = make_subplots(rows=4, cols=2, subplot_titles=['SaccharinaLatissima Error', 'LaminariaDigitata Error', 'AgarumClathratum Error', 'Algae1 Error', 'Algae2 Error', 'Rock Error', 'Unknown Error', 'OtherKelp Error'])

fig3.add_trace(go.Histogram(x=sl, name='SaccharinaLatissima'), row=1, col=1)
fig3.add_trace(go.Histogram(x=ld, name='LaminariaDigitata'), row=1, col=2)
fig3.add_trace(go.Histogram(x=ac, name='AgarumClathratum'), row=2, col=1)
fig3.add_trace(go.Histogram(x=algae1, name='Algae1'), row=2, col=2)
fig3.add_trace(go.Histogram(x=algae2, name='Algae2'), row=3, col=1)
fig3.add_trace(go.Histogram(x=rock, name='Rock'), row=3, col=2)
fig3.add_trace(go.Histogram(x=unknown, name='Unknown'), row=4, col=1)
fig3.add_trace(go.Histogram(x=ok, name='OtherKelp'), row=4, col=2)

fig3.update_layout(title_text="Absolute Error", height=1000)
go.Figure.write_html(fig3,"error_pretrain.html")