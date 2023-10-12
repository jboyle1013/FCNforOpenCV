import tensorflow as tf


def conv_block(inputs, num_filters):
    # Applying the sequence of Convolutional, Batch Normalization
    # and Activation Layers to the input tensor
    x = tf.keras.Sequential([
        # Convolutional Layer
        tf.keras.layers.Conv2D(num_filters, 3, padding='same'),
        # Batch Normalization Layer
        tf.keras.layers.BatchNormalization(),
        # Activation Layer
        tf.keras.layers.Activation('relu'),
        # Convolutional Layer
        tf.keras.layers.Conv2D(num_filters, 3, padding='same'),
        # Batch Normalization Layer
        tf.keras.layers.BatchNormalization(),
        # Activation Layer
        tf.keras.layers.Activation('relu')
    ])(inputs)

    # Returning the output of the Convolutional Block
    return x
def unet_model(input_shape=(256, 256, 3), num_classes=6, deep_supervision=True):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoding Path
    x_00 = conv_block(inputs, 64)
    x_10 = conv_block(tf.keras.layers.MaxPooling2D()(x_00), 128)
    x_20 = conv_block(tf.keras.layers.MaxPooling2D()(x_10), 256)
    x_30 = conv_block(tf.keras.layers.MaxPooling2D()(x_20), 512)
    x_40 = conv_block(tf.keras.layers.MaxPooling2D()(x_30), 1024)

    # Nested Decoding Path
    x_01 = conv_block(tf.keras.layers.concatenate(
        [x_00, tf.keras.layers.UpSampling2D()(x_10)]), 64)
    x_11 = conv_block(tf.keras.layers.concatenate(
        [x_10, tf.keras.layers.UpSampling2D()(x_20)]), 128)
    x_21 = conv_block(tf.keras.layers.concatenate(
        [x_20, tf.keras.layers.UpSampling2D()(x_30)]), 256)
    x_31 = conv_block(tf.keras.layers.concatenate(
        [x_30, tf.keras.layers.UpSampling2D()(x_40)]), 512)

    x_02 = conv_block(tf.keras.layers.concatenate(
        [x_00, x_01, tf.keras.layers.UpSampling2D()(x_11)]), 64)
    x_12 = conv_block(tf.keras.layers.concatenate(
        [x_10, x_11, tf.keras.layers.UpSampling2D()(x_21)]), 128)
    x_22 = conv_block(tf.keras.layers.concatenate(
        [x_20, x_21, tf.keras.layers.UpSampling2D()(x_31)]), 256)

    x_03 = conv_block(tf.keras.layers.concatenate(
        [x_00, x_01, x_02, tf.keras.layers.UpSampling2D()(x_12)]), 64)
    x_13 = conv_block(tf.keras.layers.concatenate(
        [x_10, x_11, x_12, tf.keras.layers.UpSampling2D()(x_22)]), 128)

    x_04 = conv_block(tf.keras.layers.concatenate(
        [x_00, x_01, x_02, x_03, tf.keras.layers.UpSampling2D()(x_13)]), 64)

    # Deep Supervision Path
    # If deep supervision is enabled, then the model will output the segmentation maps
    # at each stage of the decoding path
    if deep_supervision:
        outputs = [
            tf.keras.layers.Conv2D(num_classes, 1)(x_01),
            tf.keras.layers.Conv2D(num_classes, 1)(x_02),
            tf.keras.layers.Conv2D(num_classes, 1)(x_03),
            tf.keras.layers.Conv2D(num_classes, 1)(x_04)
        ]
        # Concatenating the segmentation maps
        outputs = tf.keras.layers.concatenate(outputs, axis=0)

    # If deep supervision is disabled, then the model will output the final segmentation map
    # which is the segmentation map at the end of the decoding path
    else:
        outputs = tf.keras.layers.Conv2D(num_classes, 1)(x_04)

    mask_head = tf.keras.layers.Conv2D(filters=6, kernel_size=1, strides=1, padding='same')(outputs)  # 1x1 convolution
    mask_head = tf.keras.layers.GlobalMaxPooling2D()(mask_head)  # Global max-pooling to reduce spatial dimensions

    # Creating the model
    model = tf.keras.Model(
        inputs=inputs, outputs=mask_head, name='Unet_plus_plus')

    # Returning the model
    return model
