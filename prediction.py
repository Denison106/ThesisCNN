import tensorflow as tf
# Or, if you want to handle Lambda layer specifically
from tensorflow.keras.layers import Lambda
# Override the deserialization with unsafe mode
from tensorflow.keras import config

from utils import test_model, load_sino_and_azim
from network import norm_vec

#sino_path = r"C:\Users\jw\Desktop\dyplomy\Erkosar Deniz\test_data\sim_sino_beads.mat"
sino_path = r"C:\Users\jw\Desktop\dyplomy\Erkosar Deniz\test_data\exp_sino.mat"
model_path =\
    r"C:\Users\jw\Desktop\dyplomy\Erkosar Deniz\Deniz code\ThesisCNN\trained_model1\model_checkpoint.keras"

sino, azim_vec = load_sino_and_azim(sino_path)

try:
    model = tf.keras.models.load_model(model_path, custom_objects={'Lambda': Lambda(norm_vec)})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

display = True
indices = list(range(0, 180, 20))
if azim_vec:
    azim_vec = azim_vec[indices]
pred_output = test_model(model, sino[:, :, indices], display, gt_azimuth=azim_vec)