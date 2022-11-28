import seaborn as sns
from tensorflow.keras.models import Sequential
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

df  = pd.read_csv('../cleaneddataset2.csv')
mlflow.tensorflow.autolog()

df[['Kelp', 'algae1', 'algae2', 'rock', 'unknown']] = df[['Kelp', 'algae1', 'algae2', 'rock', 'unknown']].fillna(0)
df[['Kelp', 'algae1', 'algae2', 'rock', 'unknown']] = df[['Kelp', 'algae1', 'algae2', 'rock', 'unknown']].div(df[['Kelp', 'algae1', 'algae2', 'rock', 'unknown']].sum(axis=1), axis=0)

scale_percent = 30
train_image = []
train_labels = []

test_image = []
test_labels = []

for i in tqdm(df.itertuples()):
    label = [i.Kelp, i.algae1, i.algae2, i.rock, i.unknown]
    island = i.SiteName.replace(" ", "")
    m = str(i[2]) + 'm'
    diver = m + '_' + i.Diver
    im_num = str(i.ImageNumber).zfill(3)
    im_name = island + str(i[2]) + i.Diver[0] + '_' + im_num


    path = "../videos/" + island + '/' + diver + '/'
    directory = os.listdir(path)


    if(island=='BarrenI' and diver=='6m_KA'):
        img3 = cv2.imread(path + str(i.ImageNumber) + '.jpg')
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        width3 = int(img3.shape[1] * scale_percent / 100)
        height3 = int(img3.shape[0] * scale_percent / 100)
        dim3 = (width3, height3)
        resized3 = cv2.resize(img3, dim3)
        train_image.append(resized3)
        train_labels.append(label)

    elif(island=='TheMoll'):
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
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                width = int(img1.shape[1] * scale_percent / 100)
                height = int(img1.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized1 = cv2.resize(img1, dim)
                test_image.append(resized1)
                test_labels.append(label)
                for q in dict1[p]:
                    img2 = cv2.imread(path + str(q) + '.jpg')
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    width2 = int(img2.shape[1] * scale_percent / 100)
                    height2 = int(img2.shape[0] * scale_percent / 100)
                    dim2 = (width2, height2)
                    resized2 = cv2.resize(img2, dim2)
                    test_image.append(resized2)
                    test_labels.append(label)

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

X_train = np.array(train_image, dtype='float32')
X_test = np.array(test_image, dtype='float32')
y_train = np.array(train_labels, dtype='float32')
y_test = np.array(test_labels, dtype='float32')

X_train = X_train/255.0
X_test = X_test/255.0

fig = make_subplots(rows=5, cols=2, subplot_titles=['Kelp Train', 'Kelp Test', 'Algae1 Train', 'Algae1 Test','Algae2 Train', 'Algae2 Test', 'Rock Train', 'Rock Test', 'Unknown Train', 'Unknown Test'])

fig.add_trace(go.Histogram(x=y_train[:,0], name='Kelp Train'), row=1, col=1)
fig.add_trace(go.Histogram(x=y_test[:,0], name='Kelp Test'), row=1, col=2)
fig.add_trace(go.Histogram(x=y_train[:,1], name='Algae1 Train'), row=2, col=1)
fig.add_trace(go.Histogram(x=y_test[:,1], name='Algae1 Test'), row=2, col=2)
fig.add_trace(go.Histogram(x=y_train[:,2], name='Algae2 Train'), row=3, col=1)
fig.add_trace(go.Histogram(x=y_test[:,2], name='ALgae2 Test'), row=3, col=2)
fig.add_trace(go.Histogram(x=y_train[:,3], name='Rock Train'), row=4, col=1)
fig.add_trace(go.Histogram(x=y_test[:,3], name='Rock Test'), row=4, col=2)
fig.add_trace(go.Histogram(x=y_train[:,4], name='Unknown Train'), row=5, col=1)
fig.add_trace(go.Histogram(x=y_test[:,4], name='Unknown Test'), row=5, col=2)

fig.update_layout(title_text="Percentage Distribution", height=1500, xaxis_title="Percent Distribution", yaxis_title="Count")
go.Figure.write_html(fig,"Distributio_molln_30.html")


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
class_weights = generate_class_weights(y_train)

class ResnetBlock(Model):
    def __init__(self, channels: int, down_sample=False):

        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)

        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

model = ResNet18(5)
model.build(input_shape = (None, 456, 811, 3))

sgd = optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd,loss=tf.keras.losses.KLDivergence(), metrics=['accuracy',tf.keras.metrics.AUC()])

early_stopping = EarlyStopping(patience=15, monitor='val_loss', verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1)
checkpoint = ModelCheckpoint("moll_test_30.ckpt", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)

history = model.fit(X_train, y_train,  use_multiprocessing=True,
                    validation_data=(X_test, y_test), batch_size=32,
                    epochs=50, class_weight=class_weights, callbacks=[early_stopping, checkpoint, reduce_lr])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('moll_loss.png')
