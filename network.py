from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import pickle
import os



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
