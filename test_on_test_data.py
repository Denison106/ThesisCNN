import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Lambda
from network import norm_vec

model_path = \
    r"C:\Users\jw\Desktop\dyplomy\Erkosar Deniz\best_trained_models\two_outputs__amplitude_no_repetitions\100epok\more50epochs\model_checkpoint.keras"


try:
    model = tf.keras.models.load_model(model_path, custom_objects={'Lambda': Lambda(norm_vec)})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

loaded = np.load('test_dataset.npz')
test_data = loaded['data']
test_labels = loaded['labels']
predicted_labels = model.predict(test_data)

# Display results
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1400)
df = pd.DataFrame({
    'Original cos(a)': test_labels[:, 0],
    'Predicted cos(a)': predicted_labels[:, 0],
    'Error cos(a)': np.abs(predicted_labels[:, 0] - test_labels[:, 0]),
    'Original sin(a)': test_labels[:, 1],
    'Predicted sin(a)': predicted_labels[:, 1],
    'Error sin(a)': np.abs(predicted_labels[:, 1] - test_labels[:, 1]),
})
print(df)
print("\n Mean values:")
print(df.mean())

for i in range(test_data.shape[0]):
    current_test_data = test_data[i, ...]
    predicted_lab = model.predict(tf.expand_dims(current_test_data, axis=0))
    plt.figure()
    plt.imshow(current_test_data, cmap="viridis")
    title_txt = "gt: ca = {:.2f}; sa = {:.2f}; a={:.2f} \n pred: ca = {:.2f}; sa = {:.2f}; a={:.2f}"
    gt_azimuth = np.rad2deg(np.arctan2(test_labels[i, 1], test_labels[i, 0]))
    pred_azimuth = np.rad2deg(np.arctan2(predicted_lab[0, 1], predicted_lab[0, 0]))
    print(test_labels[i, 1] ** 2 + test_labels[i, 0] ** 2)
    plt.title(title_txt.format(test_labels[i, 0], test_labels[i, 1], gt_azimuth,
                                     predicted_lab[0, 0], predicted_lab[0, 1], pred_azimuth))
    plt.colorbar()

plt.show()
