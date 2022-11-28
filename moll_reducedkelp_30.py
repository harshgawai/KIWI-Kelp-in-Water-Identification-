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
from time import time

df  = pd.read_csv('../cleaneddataset2.csv')

mlflow.tensorflow.autolog()

df = df.drop(['ZosteraMarina', 'CodiumFragile', 'Fucusspp', 'Turf', 'ChondusCrispus',
       'PalmariaPalmata', 'CorallineAlgae', 'Desmarestiaspp',
       'UnifentifiedAlgae', 'Bedrock', 'Boulder', 'Cobble', 'Pebble', 'Sand',
       'PlumbLine', 'Unknown', 'MixedChondrusTurf',
       'CorallinaOfficinalis', 'MixedChondrusTurfCorallina', 'MixedChondrusCoralina',
       'MixedTurfCorallina', 'TotalPts', 'Kelp'], axis=1)

df['SaccharinaLatissima']=df.iloc[:,[12,19]].sum(axis=1)
df['OtherKelp']=df.iloc[:,[15,16,17,18]].sum(axis=1)

df = df.drop(['AlariaEsculenta', 'SacchorizaDermatodea', 'UnidentifiedKelp',
       'MembraniporaMembranacea', 'JuvenileSaccharinaLatissima'], axis=1)

df.iloc[:, 12:] = df.iloc[:, 12:].fillna(0)
df.iloc[:, 12:] = df.iloc[:, 12:].div(df.iloc[:, 12:].sum(axis=1), axis=0)


train_image = []
train_labels = []
test_image = []
test_labels = []
scale_percent = 30
for i in tqdm(df.itertuples()):
    label = list(i[13:])
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

fig = make_subplots(rows=8, cols=2, subplot_titles=['SaccharinaLatissima Train', 'SaccharinaLatissima Test','LaminariaDigitata Train', 'LaminariaDigitata Test','AgarumClathratum Train', 'AgarumClathratum Test', 'Algae1 Train', 'Algae1 Test','Algae2 Train', 'Algae2 Test', 'Rock Train', 'Rock Test', 'Unknown Train', 'Unknown Test', 'OtherKelp Train', 'OtherKelp Test'])

fig.add_trace(go.Histogram(x=y_train[:,0], name='SaccharinaLatissima Train'), row=1, col=1)
fig.add_trace(go.Histogram(x=y_test[:,0], name='SaccharinaLatissima Test'), row=1, col=2)
fig.add_trace(go.Histogram(x=y_train[:,1], name='LaminariaDigitata Train'), row=2, col=1)
fig.add_trace(go.Histogram(x=y_test[:,1], name='LaminariaDigitata Test'), row=2, col=2)
fig.add_trace(go.Histogram(x=y_train[:,2], name='AgarumClathratum Train'), row=3, col=1)
fig.add_trace(go.Histogram(x=y_test[:,2], name='AgarumClathratum Test'), row=3, col=2)
fig.add_trace(go.Histogram(x=y_train[:,3], name='Algae1 Train'), row=4, col=1)
fig.add_trace(go.Histogram(x=y_test[:,3], name='Algae1 Test'), row=4, col=2)
fig.add_trace(go.Histogram(x=y_train[:,4], name='Algae2 Train'), row=5, col=1)
fig.add_trace(go.Histogram(x=y_test[:,4], name='ALgae2 Test'), row=5, col=2)
fig.add_trace(go.Histogram(x=y_train[:,5], name='Rock Train'), row=6, col=1)
fig.add_trace(go.Histogram(x=y_test[:,5], name='Rock Test'), row=6, col=2)
fig.add_trace(go.Histogram(x=y_train[:,6], name='Unknown Train'), row=7, col=1)
fig.add_trace(go.Histogram(x=y_test[:,6], name='Unknown Test'), row=7, col=2)
fig.add_trace(go.Histogram(x=y_train[:,7], name='OtherKelp Train'), row=8, col=1)
fig.add_trace(go.Histogram(x=y_test[:,7], name='OtherKelp Test'), row=8, col=2)


fig.update_layout(title_text="Percentage Distribution", height=1800)
go.Figure.write_html(fig,"Distribution_mollreducedkelp_30.html")

def generate_class_weights(class_series): #Generating class weight based on frequency of labels in the dataset
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

#Resnet Architecture
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


start2=time()


model = ResNet18(8)
model.build(input_shape = (None, 456, 811, 3))

sgd = optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd,loss=tf.keras.losses.KLDivergence(), metrics=['accuracy',tf.keras.metrics.AUC()])

early_stopping = EarlyStopping(patience=15, monitor='val_loss', verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1)
checkpoint = ModelCheckpoint("moll_reducedkelp_30.ckpt", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)


history = model.fit(X_train, y_train,  use_multiprocessing=True,
                    validation_data=(X_test, y_test),
                    epochs=50, batch_size=32,
                    class_weight=class_weights, callbacks=[early_stopping, checkpoint, reduce_lr])

end2=time()
print('Finished time: ', end2-start2)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=list(range(len(history.history['loss']))), y=history.history['loss'], name='train loss', mode='lines'))
fig2.add_trace(go.Scatter(x=list(range(len(history.history['val_loss']))), y=history.history['val_loss'], name='test loss', mode='lines'))
fig2.update_layout(xaxis_title='Epochs', yaxis_title='loss')
go.Figure.write_html(fig2,"loss_mollreducedkelp_30.html")

# model = ResNet18(8)
# model.build(input_shape = (None, 456, 811, 3))
# sgd = optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#
# model.compile(optimizer = sgd,loss=tf.keras.losses.KLDivergence(), metrics=['accuracy',tf.keras.metrics.AUC()])
# model.load_weights('tuffin_subclasskelp_10.ckpt')

loss,acc,auc = model.evaluate(X_test,  y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
proba = model.predict(X_test)


mape = mean_absolute_error(y_test, proba)
print(mape)

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
go.Figure.write_html(fig3,"error_mollreducedkelp_30.html")


