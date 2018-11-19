from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.models import save_model
from src.DatasetManager import prepare_dataset
from data_studies.GraphicsGenerator import create_comparison_graphic_with, plot_loss


def train_model(dataset_path, sample_size):
    model = create_model(sample_size)
    model.compile(loss='mean_squared_error', optimizer='Adam')

    dataframe = prepare_dataset(dataset_path, sample_size)
    x_train = dataframe.iloc[:, 0:2*sample_size].values
    x_train = (x_train - x_train.min()) / x_train.max() - x_train.min()

    y_train = dataframe.iloc[:, 2*sample_size].values.reshape(-1, 1)
    y_train = (y_train - y_train.min()) / y_train.max() - y_train.min()

    test_dataframe = prepare_dataset('./data/dataset_test.xls', sample_size)
    x_test = test_dataframe.iloc[:, 0:2*sample_size].values
    x_test = (x_test - x_test.min()) / x_test.max() - x_test.min()

    y_test = test_dataframe.iloc[:, 2*sample_size].values.reshape(-1, 1)
    y_test = (y_test - y_test.min()) / y_test.max() - y_test.min()

    lossList = []

    epochs = 600

    for i in range(0, epochs):
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        loss = model.evaluate(x_test, y_test)
        lossList.append(loss)

    create_comparison_graphic_with(y_test, model.predict(x_test))
    plot_loss(lossList)
    save_model(model, 'trained_model/model.hdf5')

def create_model(sample_size):
    model = Sequential()
    model.add(Dense(units=20, activation='relu', input_dim=2 * sample_size))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model

def test_trained_model(sample_size):
    model = get_trained_model()
    test_dataframe = prepare_dataset('./data/dataset_test.xls', sample_size)
    x_test = test_dataframe.iloc[:, 0:2 * sample_size].values
    x_test = (x_test - x_test.min()) / x_test.max() - x_test.min()

    y_test = test_dataframe.iloc[:, 2 * sample_size].values.reshape(-1, 1)
    y_test = (y_test - y_test.min()) / y_test.max() - y_test.min()

    create_comparison_graphic_with(y_test, model.predict(x_test))


def get_trained_model():
    return load_model('trained_model/model.hdf5')