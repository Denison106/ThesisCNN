from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Activation, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf


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


def hybrid_loss(y_true, y_pred):
    # Normalize vectors
    y_true_norm = tf.math.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.math.l2_normalize(y_pred, axis=-1)

    # Compute the cosine similarity (dot product)
    cosine_similarity = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)

    # Use the absolute value of cosine similarity to ignore direction signs
    cosine_loss = 1.0 - tf.abs(cosine_similarity)

    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    return 0.5 * cosine_loss + 0.0 * mse

@register_keras_serializable()
def norm_vec(y):
    return tf.linalg.l2_normalize(y, axis=1)

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

    # Convolutional layers with ReLU activation and He Normal initialization
    x = Conv2D(num_filters, filter_size, strides=2, padding='same', kernel_initializer=HeNormal())(input_layer)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    x = Conv2D(2 * num_filters, filter_size, strides=2, padding='same', kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    x = Conv2D(4 * num_filters, filter_size, strides=2, padding='same', kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    x = Conv2D(8 * num_filters, filter_size, strides=2, padding='same', kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    x = Conv2D(16 * num_filters, filter_size, strides=2, padding='same', kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    x = Conv2D(32 * num_filters, filter_size, strides=2, padding='same', kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_prob)(x)

    # Dense layers with ReLU activation and He Normal initialization
    x = Flatten()(x)  # Flatten the output from convolutional layers
    x = Dense(num_filters * 2, activation='relu', kernel_initializer=HeNormal())(x)
    x = Dropout(dropout_prob)(x)
    x = Dense(num_filters, activation='relu', kernel_initializer=HeNormal())(x)
    x = Dropout(dropout_prob)(x)

    # Output layer
    output_layer = Dense(2, activation='linear')(x)

    # Normalize output to lie on the unit circle (sin^2+cos^2 = 1)
    #normalized_output = Lambda(norm_vec)(output_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with Adam optimizer and mean absolute error loss
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=MeanAbsoluteError(),
                  metrics=['mae'])

    return model