from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

def build_model(input_shape, learning_rate):
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
        Conv2D(16, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # Predict azimuth angle directly
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=MeanSquaredError(),
                  metrics=['mae'])
    return model
