from tensorflow.keras.layers import Input, Conv2DTranspose, UpSampling2D
from confusion_met import confu_matrix,multi_confu_matrix
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, MaxPooling2D,SimpleRNN,Bidirectional,LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, confusion_matrix
import hashlib
from datetime import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def lstm(lstm_X_train, Y_train, lstm_X_test, Y_test):
    lstm_X_train = lstm_X_train.values.astype('float32')
    lstm_X_test = lstm_X_test.values.astype('float32')
    lstm_X_train = lstm_X_train.reshape(-1, 1, lstm_X_train.shape[1]).astype('float32')
    lstm_X_test = lstm_X_test.reshape(-1, 1, lstm_X_test.shape[1]).astype('float32')
    Y_train = Y_train.astype('float32')

    model = Sequential()
    model.add(LSTM(64, input_shape=lstm_X_train[0].shape, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse'])
    model.fit(lstm_X_train, Y_train, epochs=10, batch_size=64, verbose=0)
    y_pred = (model.predict(lstm_X_test) > 0.5).astype("int32")
    return y_pred, confu_matrix(Y_test, y_pred)
def autoEncoder(x_train, y_train, x_test, y_test):
    x_train = x_train.values.astype('float32')
    x_test = x_test.values.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    x_train = x_train.reshape(-1, x_train.shape[1], 1, 1)
    x_test = x_test.reshape(-1, x_test.shape[1], 1, 1)

    input_layer = Input(shape=(x_train.shape[1], 1, 1), name="INPUT")

    x = Conv2D(16, (1, 1), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((1, 1))(x)
    x = Conv2D(8, (1, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D((1, 1))(x)
    x = Conv2D(8, (1, 1), activation='relu', padding='same')(x)

    code_layer = MaxPooling2D((1, 1), name="CODE")(x)

    x = Conv2DTranspose(8, (1, 1), activation='relu', padding='same')(code_layer)
    x = UpSampling2D((1, 1))(x)
    x = Conv2DTranspose(8, (1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((1, 1))(x)
    x = Conv2DTranspose(16, (1, 1), activation='relu', padding='same')(x)
    x = UpSampling2D((1, 1))(x)

    output_layer = (Conv2D(1, (1, 1), activation='relu', padding='same', name="OUTPUT"))(x)

    a_encoder = Model(input_layer, output_layer)
    a_encoder.compile(optimizer='adam', loss='mse')
    a_encoder.fit(x_train, y_train, batch_size=8, epochs=2, shuffle=True)

    get_a_encoder = Model(inputs=a_encoder.input, outputs=a_encoder.get_layer("CODE").output)
    y_predict = np.argmax(np.argmax(get_a_encoder.predict(x_test), axis=1), axis=-1).reshape(-1)
    mcm = multi_confu_matrix(y_test, y_predict)
    return y_predict, mcm



def build_generator(latent_dim, output_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=latent_dim, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    return model

def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

def GAN(X_train, Y_train, X_test, Y_test):
    X_train = X_train.values.astype('float32')
    X_test = X_test.values.astype('float32')
    latent_dim = 10  # Number of dimensions for the generator input
    output_dim = X_train.shape[1]  # Adjust output_dim based on input feature dimension
    n_epochs = 2
    n_batch = 64

    discriminator = build_discriminator(output_dim)
    discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.fit(X_train, Y_train, epochs=10, batch_size=1200, verbose=0)

    generator = build_generator(latent_dim, output_dim)

    gan_model = build_gan(generator, discriminator)
    gan_model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

    # Generate samples for evaluation
    noise = np.random.randn(X_test.shape[0], latent_dim)
    generated_samples = generator.predict(noise)

    # Predict labels for the real and generated sample
    y_pred = (discriminator.predict(X_test) > 0.5).astype("int32")
    y_generated_pred = discriminator.predict(generated_samples)
    accuracy_generated = np.mean((y_generated_pred <= 0.5).astype(int) == 0)
    met=multi_confu_matrix(Y_test,y_pred)
    return accuracy_generated,met




def cnnlstm(X_train, y_train, X_test, y_test):
    # Convert DataFrames to numpy arrays and reshape for CNN-LSTM
    X_train = X_train.values.astype('float32')
    X_test = X_test.values.astype('float32')
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build CNN-LSTM model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test), steps_per_epoch=3)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    # Predict and generate confusion matrix
    y_predict = np.argmax(model.predict(X_test), axis=1)
    # Assuming confu_matrix is a function defined elsewhere
    met = confu_matrix(y_test, y_predict)
    return accuracy, met


def fdnn(x_train, y_train, x_test, y_test):
    # Create a Sequential model
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    model = Sequential()

    # Add an input layer with the appropriate input shape
    input_shape = x_train.shape[1]
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))

    # Add one or more hidden layers (you can add more Dense layers if needed)
    model.add(Dense(32, activation='relu'))

    # Add an output layer with the appropriate number of classes and activation function
    num_classes = len(np.unique(y_train))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with an appropriate optimizer, loss, and metrics
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=1)

    # Make predictions on the test data
    y_predict = np.argmax(model.predict(x_test), axis=1)

    # Convert y_test to int32 (assuming it's not already)
    y_test = y_test.astype('int32')

    return y_predict


def decision_tree(X_train, Y_train, X_test, Y_test):
    """
    Train a Decision Tree Classifier and make predictions.

    Parameters:
    - X_train: Training features
    - Y_train: Training labels
    - X_test: Testing features
    - Y_test: Testing labels
    - max_depth: Maximum depth of the decision tree (default is 10)

    Returns:
    - y_predict: Predicted labels for X_test
    - confusion_mat: Confusion matrix for evaluating model performance
    """
    # Create Decision Tree Classifier
    model = DecisionTreeClassifier(max_depth=10, random_state=0)

    # Train the model
    model.fit(X_train,Y_train)

    # Make predictions
    y_predict = model.predict(X_test)

    # Evaluate the model (confusion matrix example)
    confusion_mat =multi_confu_matrix(Y_test, y_predict)



    return y_predict, confusion_mat


def rnn(x_train, x_test, y_train, y_test):
    # Reshape your input data if needed
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    model = Sequential()

    # Adjust the input shape in the SimpleRNN layer
    model.add(SimpleRNN(64, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(x_train, y_train, epochs=5, batch_size=8, verbose=1)

    predicted_values = np.argmax(model.predict(x_test), axis=1)
    y_test = y_test.astype('int32')

    return predicted_values,multi_confu_matrix(y_test, predicted_values)
