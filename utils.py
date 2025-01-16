import matplotlib.pyplot as plt

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
    plt.plot(history.history['loss'], label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(history.history['mae'], label='Mean Absolute Error', color='red')
    plt.title('Mean Absolute Error Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()
