from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from tensorflow.experimental import numpy as tnp # start using tnp instead of numpy or math library
import tensorflow as tf
import pickle
import os


def build_deniz_model(input_shape, learning_rate):
    """
    Builds a simple CNN model for azimuth angle prediction.

    Parameters:
        input_shape (tuple): Shape of the input data (height, width, channels).
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        Model: Compiled Keras model.
    """

    model = Sequential([
        Input(shape=input_shape),
        Conv2D(8, (5, 5), strides=2, padding='same', activation='relu'),
        Conv2D(16, (5, 5), strides=2, padding='same', activation='relu'),
        Conv2D(32, (5, 5), strides=2, padding='same', activation='relu'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Predict azimuth angle directly
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=MeanSquaredError(),
                  metrics=['mae'])
    return model


def scaled_sigmoid_pi(x):
    return tf.constant(tnp.pi) * tf.sigmoid(x)


def build_winnik_model(input_shape, learning_rate):
    """
    Builds a simple CNN model for azimuth angle prediction.

    Parameters:
        input_shape (tuple): Shape of the input data (height, width, channels).
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        Model: Compiled Keras model.
    """

    dropout_prob = 0.1
    num_filters = 4
    filter_size = 5

    input_layer = Input(shape=input_shape)

    # Convolutional layers with ReLU activation
    x = Conv2D(num_filters, filter_size, strides=2, padding='same')(input_layer)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    x = Conv2D(2 * num_filters, filter_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    x = Conv2D(4 * num_filters, filter_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    x = Conv2D(8 * num_filters, filter_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    x = Conv2D(16 * num_filters, filter_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    x = Conv2D(32 * num_filters, filter_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    # Dense layers with droputs
    x = Flatten()(x)  # make it 1D data
    x = Dense(num_filters * 2, activation='relu')(x)
    x = Dropout(dropout_prob)(x)
    x = Dense(num_filters, activation='relu')(x)
    x = Dropout(dropout_prob)(x)
    #output_layer = Dense(1, activation=lambda x: tf.constant(tnp.pi) * tf.sigmoid(x))(x)
    output_layer = Dense(1, activation=scaled_sigmoid_pi)(x)
    #output_layer = Dense(1, activation='linear')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=MeanAbsoluteError(),
                  metrics=['mae'])
    return model


def build_yutaro_model(input_shape, learning_rate):
    """
    Builds a simple CNN model for azimuth angle prediction.

    Parameters:
        input_shape (tuple): Shape of the input data (height, width, channels).
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        Model: Compiled Keras model.
    """

    dropout_prob = 0.1
    num_filters = 64
    filter_size = 5
    input_layer = Input(shape=input_shape)
    x = Conv2D(num_filters, filter_size, strides=2, padding='same')(input_layer)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, filter_size, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters * 2, filter_size, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters * 2, filter_size, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(num_filters * 2, activation='relu')(x)
    x = Dropout(dropout_prob)(x)
    x = Dense(num_filters, activation='relu')(x)
    x = Dropout(dropout_prob)(x)
    x = Dense(1)(x)
    output_layer = Activation('sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=MeanAbsoluteError(),
                  metrics=['mae'])

    return model


def build_mc_model(input_shape, learning_rate):
    """
    Builds a simple CNN model for azimuth angle prediction.

    Parameters:
        input_shape (tuple): Shape of the input data (height, width, channels).
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        Model: Compiled Keras model.
    """
    dropout_prob = 0.5
    num_filters = 32#64
    filter_size = 5
    input_layer = Input(shape=input_shape)

    x = Conv2D(num_filters, filter_size, strides=2, padding='same')(input_layer)
    x = Dropout(dropout_prob)(x)
    x = Activation('relu')(x)

    x = Conv2D(2 * num_filters, filter_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(4 * num_filters, filter_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(8 * num_filters, filter_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(16 * num_filters, filter_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32 * num_filters, filter_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    output_layer = Conv2D(1, 4, activation='linear', name='regressionoutput')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=MeanSquaredError(),
                  metrics=['mae'])
    return model


def prepare_callbacks():
    # Model checkpoint callback for saving weights
    tensorboard = TensorBoard(
        os.path.join(os.getcwd(), "logs"),
        histogram_freq=1,
        write_steps_per_second=True,
        write_images=True,
        update_freq='epoch'
    )

    # Define the subdirectory path
    subdir = os.path.join(os.getcwd(), "trained_model")
    os.makedirs(subdir, exist_ok=True)
    # Model checkpoint callback for saving weights
    checkpoint = ModelCheckpoint(
        os.path.join(os.getcwd(), r'trained_model/epoch_{epoch:02d}_model_checkpoint.keras'),
        save_freq="epoch"
    )

    class SaveBatchLoss(Callback):
        def on_train_begin(self, logs={}):
            self.train_losses = []

        def on_train_batch_end(self, batch, logs={}):
            self.train_losses.append(logs.get('loss'))

        def on_train_end(self, logs={}):
            with open("trained_model/train_losses", "wb") as fp:  # pickling
                pickle.dump(self.train_losses, fp)
    save_batch_loss = SaveBatchLoss()

    return [save_batch_loss, checkpoint]
