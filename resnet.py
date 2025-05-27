"""
ResNet-20 model
There are 3 groups. Each group has n=3 residual blocks. Each residual block has 2 Conv2D layers.
This relates to total number of 3*2*n + 2 = 20 layers.
The authors claim that the code worked only for SGD, not Adam or SGDW!
source:
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-resnet-from-scratch-with-tensorflow-2-and-keras.md
how to get Tesnorboard type in terminal: python -m tensorboard.main --logdir=logs/
"""
import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, \
    Conv2D, Lambda, Input, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay


def define_config(learning_rate):
    # Generic configuration
    verbose = 1
    n = 3  # number of residual blocks in a single group
    init_fm_dim = 16  # initial number of feature maps; doubles as the feature map size halves
    shortcut_type = "identity"  # shortcut type: "identity" or "projection"

    # Define the loss function
    loss = tensorflow.keras.losses.MeanSquaredError()

    # Set layer initializer
    initializer = tensorflow.keras.initializers.HeNormal()

    # Define the optimizer
    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate,#1e-5,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = Adam(learning_rate=lr_schedule)

    # Create configuration dictionary
    config = {
        # "epochs": epochs,
        # "width": width,
        # "height": height,
        # "dim": channels,
        # "batch_size": batch_size,
        # "validation_split": validation_split,
        "verbose": verbose,
        "stack_n": n,
        "initial_num_feature_maps": init_fm_dim,
        # "training_ds_size": train_size,
        # "steps_per_epoch": steps_per_epoch,
        # "val_steps_per_epoch": val_steps_per_epoch,
        # "num_epochs": epochs,
        "loss": loss,
        "optim": optimizer,
        "initializer": initializer,
        "shortcut_type": shortcut_type
    }
    return config


# def residual_block(x, number_of_filters, config):
#     """
#     Residual block with
#     """
#     initializer = config["initializer"]
#
#     # Create skip connection
#     x_skip = x
#
#     # Perform the original mapping
#     x = Conv2D(number_of_filters, kernel_size=(3, 3), strides=(1, 1),
#                kernel_initializer=initializer, padding="same")(x_skip)
#     x = BatchNormalization(axis=3)(x)
#     x = Activation("relu")(x)
#     x = Conv2D(number_of_filters, kernel_size=(3, 3),
#                kernel_initializer=initializer, padding="same")(x)
#     x = BatchNormalization(axis=3)(x)
#
#     # Add the skip connection to the regular mapping
#     x = Add()([x, x_skip])
#
#     # Nonlinearly activate the result
#     x = Activation("relu")(x)
#
#     return x


def residual_block(x, number_of_filters, config, match_filter_size=False):
    """
        Residual block with
    """
    initializer = config.get("initializer")

    # Create skip connection
    x_skip = x

    # Perform the original mapping
    if match_filter_size:
        x = Conv2D(number_of_filters, kernel_size=(3, 3), strides=(2,2),\
            kernel_initializer=initializer, padding="same")(x_skip)
    else:
        x = Conv2D(number_of_filters, kernel_size=(3, 3), strides=(1,1),\
            kernel_initializer=initializer, padding="same")(x_skip)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    x = Conv2D(number_of_filters, kernel_size=(3, 3),\
        kernel_initializer=initializer, padding="same")(x)
    x = BatchNormalization(axis=3)(x)

    # Perform matching of filter numbers if necessary
    if match_filter_size and config.get("shortcut_type") == "identity":
        x_skip = Lambda(lambda x: tensorflow.pad(x[:, ::2, ::2, :], tensorflow.constant([[0, 0,], [0, 0], [0, 0], [number_of_filters//4, number_of_filters//4]]), mode="CONSTANT"))(x_skip)
    elif match_filter_size and config.get("shortcut_type") == "projection":
        x_skip = Conv2D(number_of_filters, kernel_size=(1,1),\
            kernel_initializer=initializer, strides=(2,2))(x_skip)

    # Add the skip connection to the regular mapping
    x = Add()([x, x_skip])

    # Nonlinearly activate the result
    x = Activation("relu")(x)

    # Return the result
    return x


# def ResidualBlocks(x, config):
#     """
#     Set up the residual blocks.
#     """
#
#     # Set initial filter size
#     filter_size = config.get("initial_num_feature_maps")
#
#     # Paper: "Then we use a stack of 6n layers (...)
#     #	with 2n layers for each feature map size."
#     # 6n/2n = 3, so there are always 3 groups.
#     for layer_group in range(3):
#         # Each block in our code has 2 weighted layers,
#         # and each group has 2n such blocks,
#         # so 2n/2 = n blocks per group.
#         for block in range(config.get("stack_n")):
#             x = residual_block(x, filter_size, config)
#     # Return final layer
#     return x


def ResidualBlocks(x, config):
    """
    Set up the residual blocks.
    """
    # Retrieve values

    # Set initial filter size
    filter_size = config.get("initial_num_feature_maps")

    # Paper: "Then we use a stack of 6n layers (...)
    #	with 2n layers for each feature map size."
    # 6n/2n = 3, so there are always 3 groups.
    for layer_group in range(3):

        # Each block in our code has 2 weighted layers,
        # and each group has 2n such blocks,
        # so 2n/2 = n blocks per group.
        for block in range(config.get("stack_n")):

            # Perform filter size increase at every
            # first layer in the 2nd block onwards.
            # Apply Conv block for projecting the skip
            # connection.
            if layer_group > 0 and block == 0:
                filter_size *= 2
                x = residual_block(x, filter_size, config, match_filter_size=True)
            else:
                x = residual_block(x, filter_size, config)

    # Return final layer
    return x


def model_base(input_shape, config, path_num=2):
    # """
    # Base structure of the model, with residual blocks
    # attached.
    # """
    # initializer = config["initializer"]
    # inputs = Input(shape=input_shape)
    # x = Conv2D(config.get("initial_num_feature_maps"), kernel_size=(3, 3),
    #            strides=(1, 1), kernel_initializer=initializer, padding="same")(inputs)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    #
    # if path_num <1:
    #     print("error; number of path has to be at least 1")
    # elif path_num == 1:
    #     x = ResNetPath(x, config, path_num)
    # else:
    #     path_list=[]
    #     for path_no in range(1,path_num+1):
    #         path_list.append(ResNetPath(x, config, path_no))
    #     x = Add()(path_list)
    # x = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),
    #            kernel_initializer=initializer, padding="same")(x)
    # x = BatchNormalization()(x)
    # x = Flatten()(x)  # Flatten the output from convolutional layers
    # x = Dense(8, activation='relu', kernel_initializer=HeNormal())(x)
    # #x = Dropout(dropout_prob)(x)
    # x = Dense(2, activation='relu', kernel_initializer=HeNormal())(x)
    # #x = Dropout(dropout_prob)(x)
    # outputs = Dense(2, activation='linear')(x)
    #
    # return inputs, outputs

    # Get number of classes from model configuration
    initializer = config.get("initializer")

    # Define model structure
    # logits are returned because Softmax is pushed to loss function.
    inputs = Input(shape=input_shape)
    x = Conv2D(config.get("initial_num_feature_maps"), kernel_size=(3, 3),\
               strides=(1, 1), kernel_initializer=initializer, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = ResidualBlocks(x, config)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    outputs = Dense(2, kernel_initializer=initializer, activation='linear')(x)

    return inputs, outputs


def build_resnet(input_shape, learning_rate):
    config = define_config(learning_rate)
    inputs, outputs = model_base(input_shape, config, path_num=1)

    # Initialize and compile model
    model = Model(inputs, outputs, name=config.get("name"))

    model.compile(loss=config.get("loss"),
                  optimizer=config.get("optim"),
                  metrics=config.get("optim_additional_metrics"))

    model.summary()
    return model

