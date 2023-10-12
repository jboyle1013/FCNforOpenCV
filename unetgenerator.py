import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf


class AnnotatedImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, batch_size=16, image_size=(256, 256)):
        self.image_dir = image_dir
        self.annotation_csv = os.path.join(image_dir, "_annotations.csv")
        self.batch_size = batch_size
        self.image_size = image_size
        # Read the annotation CSV file
        self.annotations = pd.read_csv(self.annotation_csv)
        self.label_map = self.create_label_map()  # Create a mapping from string labels to integer labels
        self.num_classes = len(self.label_map)


    def create_label_map(self):
        label_map = {'d10': 3, 'd12': 4, 'd20': 5, 'd4': 0, 'd6': 1, 'd8': 2}
        # Convert the lists in the dictionary values to NumPy arrays



        return label_map



    def __len__(self):
        # Return the number of batches
        return int(np.ceil(len(self.annotations) / self.batch_size))

    def __getitem__(self, index):
        # Get the image, annotation, and one-hot encoded labels for the current batch
        image_batch = []
        annotation_batch = []
        label_batch = []

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            if i >= len(self.annotations):
                break
            image = cv2.imread(os.path.join(self.image_dir, self.annotations.iloc[i]['filename']))
            image = cv2.resize(image, self.image_size)
            annotation = [
                self.annotations.iloc[i]['width'],
                self.annotations.iloc[i]['height'],
                self.annotations.iloc[i]['xmin'],
                self.annotations.iloc[i]['ymin'],
                self.annotations.iloc[i]['xmax'],
                self.annotations.iloc[i]['ymax']
            ]
            label = self.label_map[self.annotations.iloc[i]['class']]  # Map the string label to an integer label

            image_batch.append(image)
            annotation_batch.append(annotation)
            label_batch.append(label)

        # Convert the image and annotation to numpy arrays
        image_batch = np.array(image_batch)
        annotation_batch = np.array(annotation_batch)

        label_batch = tf.one_hot(indices=label_batch, depth=1)  # Convert to one-hot encoding

        # Return the image, annotation, and one-hot encoded label batch
        return image_batch, annotation_batch, label_batch
