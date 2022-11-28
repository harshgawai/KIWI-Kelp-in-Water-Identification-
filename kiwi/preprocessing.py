import cv2
import pandas as pd
import os
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def clean(data):
    data[['Kelp', 'algae1', 'algae2', 'rock', 'unknown']] = data[['Kelp', 'algae1', 'algae2', 'rock', 'unknown']].fillna(0)
    data[['Kelp', 'algae1', 'algae2', 'rock', 'unknown']] = data[['Kelp', 'algae1', 'algae2', 'rock', 'unknown']].div(data[['Kelp', 'algae1',
                                                                                    'algae2', 'rock', 'unknown']].sum(axis=1), axis=0)
    return data

def scale(img, scale_percent):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim)
    return resized

def list_to_array(train_image, test_image, train_labels, test_labels):
    X_train = np.array(train_image, dtype='float32')
    X_test = np.array(test_image, dtype='float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = np.array(train_labels, dtype='float32')
    y_test = np.array(test_labels, dtype='float32')

    return X_train, X_test, y_train, y_test

def plot_distribution(y_train, y_test):
    fig = make_subplots(rows=5, cols=2,
                        subplot_titles=['Kelp Train', 'Kelp Test', 'Algae1 Train', 'Algae1 Test', 'Algae2 Train',
                                        'Algae2 Test', 'Rock Train', 'Rock Test', 'Unknown Train', 'Unknown Test'])

    fig.add_trace(go.Histogram(x=y_train[:, 0], name='Kelp Train'), row=1, col=1)
    fig.add_trace(go.Histogram(x=y_test[:, 0], name='Kelp Test'), row=1, col=2)
    fig.add_trace(go.Histogram(x=y_train[:, 1], name='Algae1 Train'), row=2, col=1)
    fig.add_trace(go.Histogram(x=y_test[:, 1], name='Algae1 Test'), row=2, col=2)
    fig.add_trace(go.Histogram(x=y_train[:, 2], name='Algae2 Train'), row=3, col=1)
    fig.add_trace(go.Histogram(x=y_test[:, 2], name='ALgae2 Test'), row=3, col=2)
    fig.add_trace(go.Histogram(x=y_train[:, 3], name='Rock Train'), row=4, col=1)
    fig.add_trace(go.Histogram(x=y_test[:, 3], name='Rock Test'), row=4, col=2)
    fig.add_trace(go.Histogram(x=y_train[:, 4], name='Unknown Train'), row=5, col=1)
    fig.add_trace(go.Histogram(x=y_test[:, 4], name='Unknown Test'), row=5, col=2)

    fig.update_layout(title_text="Percentage Distribution", height=1500, xaxis_title="Percent Distribution",
                      yaxis_title="Count")
    go.Figure.write_html(fig, "Distribution.html")

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
# def handle(i, test_island):
#     label = [i.Kelp, i.algae1, i.algae2, i.rock, i.unknown]
#     island = i.SiteName.replace(" ", "")
#     m = str(i[2]) + 'm'
#     diver = m + '_' + i.Diver
#     im_num = str(i.ImageNumber).zfill(3)
#     im_name = island + str(i[2]) + i.Diver[0] + '_' + im_num
#
#     path = "../../../videos/" + island + '/' + diver + '/'
#     directory = os.listdir(path)
#
#     if (island == 'BarrenI' and diver == '6m_KA'):
#         img = cv2.imread(path + str(i.ImageNumber) + '.jpg')
#         resized = scale(img, scale_percent=10)
#         return (resized, label)
#
#     else:


