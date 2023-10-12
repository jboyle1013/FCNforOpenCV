import tensorflow as tf
from model import FCN_model
from generator import Generator
import os
import matplotlib.pyplot as plt
from export_savedmodel import export, testagainstimagewmodel
from newmodel import NewFCN_model
from maskishmodel import UNet_MaskRNN
from mobilenetModel import MobileNet_Custom_Model
from unetgenerator import AnnotatedImageGenerator
from UNETPLUSPLUS import unet_model

def train(model, train_generator, val_generator, epochs=50):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics='accuracy')

    checkpoint_path = './snapshots'
    os.makedirs(checkpoint_path, exist_ok=True)
    check_path = checkpoint_path + '/' + 'model_epoch_{epoch:02d}.keras'
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/model.{epoch:02d}-{val_loss:.2f}.keras'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=my_callbacks,
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        shuffle=True)
    return history, model


if __name__ == "__main__":
    # Create FCN model
    model = UNet_MaskRNN()

    # The below folders are created using utils.py
    train_dir = 'annotated_dice/train'
    val_dir = 'annotated_dice/valid'

    # If you get out of memory error try reducing the batch size
    BATCH_SIZE = 2
    val_generator = AnnotatedImageGenerator(val_dir, BATCH_SIZE)
    train_generator = AnnotatedImageGenerator(train_dir, BATCH_SIZE)

    EPOCHS = 20
    history, model = train(model, train_generator, val_generator, epochs=EPOCHS)

    # Accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    model.save("DiceFCN.keras")
    model.save_weights(
        "DiceFCNWeights", overwrite=True, save_format=None, options=None
    )
    testagainstimagewmodel(model)
    export("DiceFCN.keras", "DiceFCNTF")
    # epochs
    epochs_range = range(EPOCHS)
    # Plotting graphs
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("graph.png")
    plt.show()
