import keras
from keras.datasets import mnist
from keras.utils.vis_utils import plot_model

def dataloader(dataset):
    if dataset.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_train = x_train.astype('float32')
        x_train /= 255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
        return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = dataloader("mnist")

    dnn_model = keras.models.load_model("./models/dnn.h5")
    plot_model(dnn_model, to_file='./models/dnn.png', show_shapes=True)
    cnn_model = keras.models.load_model("./models/cnn.h5")
    plot_model(cnn_model, to_file='./models/cnn.png', show_shapes=True)
    rnn_model = keras.models.load_model("./models/rnn.h5")
    plot_model(rnn_model, to_file='./models/rnn.png', show_shapes=True)

    res_dnn = dnn_model.predict(x_test[0:5])
    res_cnn = cnn_model.predict(x_test[0:5])
    res_rnn = rnn_model.predict(x_test[0:5])
