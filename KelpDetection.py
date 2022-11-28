from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Layer, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from skimage.transform import rotate
from sklearn.metrics import hamming_loss
import cv2

df = pd.read_csv('cleaneddataset.csv')
balance_y = pd.DataFrame(columns=['Kelp', 'algae1', 'algae2', 'rock', 'unknown'])
colorspace_y = pd.DataFrame(columns=['Kelp', 'algae1', 'algae2', 'rock', 'unknown'])

train_image = []
test_image = []
balance_x = []
colorspace_x = []

for i in tqdm(df.itertuples()):
    island = i.SiteName.replace(" ", "")
    m = str(i[2]) + 'm'
    diver = m + '_' + i.Diver
    im_num = str(i.ImageNumber).zfill(3)
    im_name = island + str(i[2]) + i.Diver[0] + '_' + im_num
    img1 = cv2.imread('EasternShoreIslandsKelpSurvey2018/' + island + "/" + diver + "/" + im_name +'.jpg')
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    scale_percent = 80
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (512, 512)
    resized = cv2.resize(img, dim)

    if(i.rock != 0 or i.unknown != 0):
        hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv = cv2.resize(hsv, dim)
        hsv[0, :, :] = hsv[0, :, :]/180.
        hsv[1:3, :, :] = hsv[1:3, :, :]/255.

        ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
        ycrcb = cv2.resize(ycrcb, dim)
        ycrcb = ycrcb/255.0

        luv = cv2.cvtColor(img1, cv2.COLOR_BGR2LUV)
        luv = cv2.resize(luv, dim)
        luv = luv/255.0


        balance_x.append(resized)

        colorspace_x.append(hsv)
        colorspace_x.append(ycrcb)
        colorspace_x.append(luv)

        for k in range(3):
            colorspace_y.loc[len(colorspace_y)] = [i.Kelp, i.algae1, i.algae2, i.rock, i.unknown]
        balance_y.loc[len(balance_y)] = [i.Kelp, i.algae1, i.algae2, i.rock, i.unknown]

    train_image.append(resized)

X = np.array(train_image, dtype='float32')
balanced = np.array(balance_x, dtype='float32')
colorspace = np.array(colorspace_x, dtype='float32')

X = X/255.0
balanced = balanced/255.

y_train = df[['Kelp', 'algae1', 'algae2', 'rock', 'unknown']]
y_train = y_train.fillna(0)
balance_y = balance_y.fillna(0)
colorspace_y = colorspace_y.fillna(0)

y_train = y_train.astype(bool).astype(int)
balance_y = balance_y.astype(bool).astype(int)
colorspace_y = colorspace_y.astype(bool).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.20, random_state=2, stratify=y_train)
ytrain = np.array(y_train)
ytest = np.array(y_test)

ybalance = np.array(balance_y)
ycolor = np.array(colorspace_y)

positive_weights = {}
negative_weights = {}
for c in y_train.columns:
    positive_weights[c] = y_train.shape[0]/(2.0*np.count_nonzero(y_train[c]==1))
    negative_weights[c] = y_train.shape[0]/(2.0*np.count_nonzero(y_train[c]==0))

def generate_class_weights(class_series):
    mlb = None
    n_samples = len(class_series)
    n_classes = len(class_series[0])

    # Count each class frequency
    class_count = [0] * n_classes
    for classes in class_series:
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1

    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
    return dict(zip(class_labels, class_weights))
class_weights = generate_class_weights(ytrain)

early_stopping = EarlyStopping(patience=5, monitor='val_loss', verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1)
checkpoint = ModelCheckpoint("cnn.ckpt", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

def build_model():
    model = Sequential()

    model.add(Conv2D(32,(3,3),padding='same',input_shape=(512,512,3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32,(3,3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(3, 3)))


    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(5, activation='sigmoid'))

    return model

sgd = optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model = build_model()
model.compile(optimizer=sgd,loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])
balanced_train = []
balanced_target = []
for i in tqdm(range(balanced.shape[0])):
    # balanced_train.append(balanced[i])
    balanced_train.append(rotate(balanced[i], angle=30, mode = 'wrap'))
    balanced_train.append(np.fliplr(balanced[i]))
    balanced_train.append(np.flipud(balanced[i]))
    for j in range(3):
        balanced_target.append(ybalance[i])

balanced_x = np.array(balanced_train)
ybalance = np.array(balanced_target)
Xtrain = np.concatenate((X_train, balanced_x, colorspace), axis=0)
target = np.concatenate((ytrain, ybalance, ycolor), axis=0)

print(len(Xtrain), len(target))
history = model.fit(Xtrain, target,  use_multiprocessing=True,
                    validation_data=(X_test, ytest),
                    epochs=50, class_weight=class_weights, callbacks=[early_stopping, checkpoint, reduce_lr])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')