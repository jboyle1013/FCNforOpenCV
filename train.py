import tensorflow as tf
from model import FCN_model
from generator import Generator
import os
import matplotlib.pyplot as plt


def train(model, train_generator, val_generator, epochs=50):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_path = './snapshots'
    os.makedirs(checkpoint_path, exist_ok=True)
    model_path = checkpoint_path + '/' + 'model_epoch_{epoch:02d}.keras'

    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                         validation_data=val_generator,
                        validation_steps=len(val_generator))
    model.save("DiceFCN.keras")
    return history


if __name__ == "__main__":
    # Create FCN model
    model = FCN_model()

    # The below folders are created using utils.py
    train_dir = 'dice/train'
    val_dir = 'dice/valid'

    # If you get out of memory error try reducing the batch size
    BATCH_SIZE = 8
    train_generator = Generator(train_dir, BATCH_SIZE, shuffle_images=True, image_min_side=24)
    val_generator = Generator(val_dir, BATCH_SIZE, shuffle_images=True, image_min_side=24)
    plt.figure(figsize=(20, 20))

    EPOCHS = 20
    history = train(model, train_generator, val_generator, epochs=EPOCHS)

    # Accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']

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
    plt.show()
