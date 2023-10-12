import tensorflow as tf
from keras.applications import MobileNet

def MobileNet_Custom_Model(len_classes=6):
    # Load MobileNet with pre-trained ImageNet weights
    base_model = MobileNet(input_shape=(None, None, 3), include_top=False, weights='imagenet')

    # Freeze MobileNet layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of MobileNet
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(len_classes, activation='softmax')(x)

    # Create the new model
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    return model

if __name__ == "__main__":
    model = MobileNet_Custom_Model(len_classes=6)
    print(model.summary())
