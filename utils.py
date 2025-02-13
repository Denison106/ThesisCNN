import matplotlib.pyplot as plt
import pickle
import numpy as np

def visualize_data(input_images, display_params):
    """
    Visualizes the simulated fringe pattern data. 
    Parameters:
        input_images (ndarray): Array of simulated fringe pattern images.
        display_params (dict): Display-related parameters.
    """
    plt.ion()
    for i in range(input_images.shape[0]):
        plt.imshow(input_images[i, :, :, 0], cmap="viridis")
        plt.colorbar()
        plt.show()
        plt.pause(display_params["display_pause_time"])
        plt.clf()
    plt.ioff()


def plot_learning_curves(history):
    """
    Plots the learning curves (loss and MAE) from the training history.

    Parameters:
        history: Training history object returned by the Keras model's `fit` method.
    """
    plt.figure()

    try:
        # Try to access and plot loss per iteration
        with open("trained_model/train_losses", "rb") as fp:
            train_loss_per_batch = pickle.load(fp)
        iters = range(1, len(train_loss_per_batch) + 1)
        plt.plot(iters, train_loss_per_batch, 'b', label='Training Loss')
    except:
        print("An exception occurred")


    steps_per_epoch = (len(train_loss_per_batch)+1)/(len(history.history['loss']) + 1)
    epochs = np.array(range(len(history.history['loss'])))+1
    plt.plot(epochs * steps_per_epoch, history.history['loss'], 'bo', label='Loss')
    plt.plot(epochs * steps_per_epoch, history.history['val_loss'], 'ro', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(history.history['mae'], label='Mean Absolute Error', color='red')
    plt.title('Mean Absolute Error Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()
