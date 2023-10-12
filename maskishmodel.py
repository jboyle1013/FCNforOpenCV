import tensorflow as tf



def UNet_MaskRNN(len_classes=6, dropout_rate=0.2):
    # Define the model's input layer
    input = tf.keras.layers.Input(shape=(None, None, 3))  # Input layer for images with variable dimensions

    # Encoding Path (similar to U-Net)

    # Block 1
    x1 = tf.keras.layers.Conv2D(16, 3, padding='same')(input)  # 2D convolution with 32 filters and 3x3 kernel
    x1 = tf.keras.layers.BatchNormalization()(x1)  # Batch normalization for training stability
    x1 = tf.keras.layers.Activation('relu')(x1)  # Rectified Linear Unit (ReLU) activation
    x1 = tf.keras.layers.Conv2D(16, 3, padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation('relu')(x1)
    pool1 = tf.keras.layers.MaxPooling2D()(x1)  # Max-pooling to downsample

    # Block 2
    x2 = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding='same')(pool1)  # Strided convolution for downsampling
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Activation('relu')(x2)
    x2 = tf.keras.layers.Conv2D(32, 3, padding='same')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Activation('relu')(x2)
    pool2 = tf.keras.layers.MaxPooling2D()(x2)  # Max-pooling to downsample

    # Continue adding more encoding blocks as needed

    # Decoding Path (similar to U-Net)

    # Upsample and concatenate
    up1 = tf.keras.layers.Conv2DTranspose(32, (2, 2), padding='same')(x2)  # Transposed convolution for upsampling
    up1 = tf.keras.layers.Concatenate()([up1, x2])  # Concatenate with the corresponding encoding block
    x3 = tf.keras.layers.Conv2D(16, 3, padding='same')(up1)  # 2D convolution with 32 filters and 3x3 kernel
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Activation('relu')(x3)
    x3 = tf.keras.layers.Conv2D(16, 3, padding='same')(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Activation('relu')(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.ReLU()(x3)  # Customized ReLU activation

    # Continue adding more decoding blocks as needed

    # Mask Head (Mask R-CNN-inspired)
    mask_head = tf.keras.layers.Conv2D(len_classes, 1, activation="Softmax")(x3)  # 1x1 convolution with softmax activation

    # Additional processing in the mask head
    mask_head = tf.keras.layers.Conv2D(filters=len_classes, kernel_size=1, strides=1, padding='same')(mask_head)  # 1x1 convolution
    mask_head = tf.keras.layers.GlobalMaxPooling2D()(mask_head)  # Global max-pooling to reduce spatial dimensions


    # Create the final model
    model = tf.keras.Model(inputs=input, outputs=mask_head)

    # Print model summary and total number of layers
    print(model.summary())
    print(f'Total number of layers: {len(model.layers)}')

    return model


# Entry point
if __name__ == "__main__":
    UNet_MaskRNN(len_classes=6, dropout_rate=0.2)