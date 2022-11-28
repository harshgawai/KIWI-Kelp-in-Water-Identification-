from Resnet18_model import create_model
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def generate_absolute_error(X_test, y_test):
    model = create_model()
    model.load_weights(model)
    proba = model.predict(X_test)

    kelp = []
    algae1 = []
    algae2 = []
    rock = []
    unknown = []
    avrg = []
    for i, j in zip(proba, y_test):
        k_mae = np.round(abs(i[0] - j[0]), 3)
        a1_mae = np.round(abs(i[1] - j[1]), 3)
        a2_mae = np.round(abs(i[2] - j[2]), 3)
        r_mae = np.round(abs(i[3] - j[3]), 3)
        u_mae = np.round(abs(i[4] - j[4]), 3)
        avg = (k_mae + a1_mae + a2_mae + r_mae + u_mae) / 5

        kelp.append(k_mae)
        algae1.append(a1_mae)
        algae2.append(a2_mae)
        rock.append(r_mae)
        unknown.append(u_mae)
        avrg.append(avg)
    k = sum(kelp) / len(kelp)
    a1 = sum(algae1) / len(algae1)
    a2 = sum(algae2) / len(algae2)
    r = sum(rock) / len(rock)
    u = sum(unknown) / len(unknown)
    print(k, a1, a2, r, u)

    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=['Kelp Error', 'Algae1 Error', 'Algae2 Error', 'Rock Error', 'Unknown Error'])

    fig.add_trace(go.Histogram(x=kelp, name='Kelp'), row=1, col=1)
    fig.add_trace(go.Histogram(x=algae1, name='Algae1'), row=1, col=2)
    fig.add_trace(go.Histogram(x=algae2, name='Algae2'), row=2, col=1)
    fig.add_trace(go.Histogram(x=rock, name='Rock'), row=2, col=2)
    fig.add_trace(go.Histogram(x=unknown, name='Unknown'), row=3, col=1)

    fig.update_layout(title_text="Absolute Error", height=1000, xaxis_title="Number of samples",
                      yaxis_title="Absolute Error")
    go.Figure.write_html(fig, "error.html")

def generate_loss_plot(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(history.history['loss']))), y=history.history['loss'], name='train loss',
                             mode='lines'))
    fig.add_trace(
        go.Scatter(x=list(range(len(history.history['val_loss']))), y=history.history['val_loss'], name='test loss',
                   mode='lines'))
    fig.update_layout(xaxis_title='Epochs', yaxis_title='loss')
    go.Figure.write_html(fig, "loss.html")