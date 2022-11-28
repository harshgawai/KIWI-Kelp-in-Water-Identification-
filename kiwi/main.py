import pandas as pd
from preprocessing import clean, scale, list_to_array, plot_distribution, generate_class_weights
from Resnet18_model import train_model, create_model
from model_evaluation import generate_loss_plot, generate_absolute_error
from pprint import pprint
from tqdm import tqdm
import cv2
import os

data = pd.read_csv('../../cleaneddataset2.csv')
df = clean(data)

test_islands = [i.replace(" ", "") for i in set(df['SiteName'])]

dict1 = {i+1:j for i,j in enumerate(test_islands)}
pprint(dict1)

num = int(input('Enter the Island number for model validation:'))
test_island = dict1[num]

# train_images, test_images, train_labels, test_labels = zip(*[handle(i, test_island) for i in tqdm(df.itertuples())])
train_image = []
train_labels = []

test_image = []
test_labels = []
scale_percent = 10
for i in tqdm(df.itertuples()):
    label = [i.Kelp, i.algae1, i.algae2, i.rock, i.unknown]
    island = i.SiteName.replace(" ", "")
    m = str(i[2]) + 'm'
    diver = m + '_' + i.Diver
    im_num = str(i.ImageNumber).zfill(3)
    im_name = island + str(i[2]) + i.Diver[0] + '_' + im_num


    path = "../../videos/" + island + '/' + diver + '/'
    directory = os.listdir(path)


    if(island=='BarrenI' and diver=='6m_KA'):
        img3 = cv2.imread(path + str(i.ImageNumber) + '.jpg')
        resized3 = scale(img3, scale_percent)
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
                resized1 = scale(img1, scale_percent)
                test_image.append(resized1)
                test_labels.append(label)
                for q in dict1[p]:
                    img2 = cv2.imread(path + str(q) + '.jpg')
                    resized2 = scale(img2, scale_percent)
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
                img4 = cv2.imread(path + p + '.jpg')
                resized4 = scale(img4, scale_percent)
                train_image.append(resized4)
                train_labels.append(label)
                for q in dict1[p]:
                    img5 = cv2.imread(path + str(q) + '.jpg')
                    resized5 = scale(img5, scale_percent)
                    train_image.append(resized5)
                    train_labels.append(label)


X_train, X_test, y_train, y_test = list_to_array(train_image, test_image, train_labels, test_labels)
plot_distribution(y_train, y_test)

class_weights = generate_class_weights(y_train)

model = create_model()
history = train_model(model, X_train, X_test, y_train, y_test, class_weights)

generate_loss_plot(history)
generate_absolute_error('themoll.ckpt', X_test, y_test)