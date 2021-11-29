import os
import argparse
import matplotlib.pyplot as plt
from typing import List

import tensorflow as tf
from tensorflow.python.data import Dataset, AUTOTUNE
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
from model import build_pretrained_model, build_augment_layer, train_model, evaluate_model, save_model

IMAGE_SIZE = 229, 229
BATCH_SIZE = 32

def get_datasets(train_directory: str, val_directory: str, test_directory: str, val_rate: float):
    if val_directory == '':
        train_ds = image_dataset_from_directory(
            directory = train_directory,
            labels = 'inferred',
            label_mode = 'categorical',
            shuffle = True,
            batch_size = BATCH_SIZE,
            image_size = IMAGE_SIZE,
            validation_split=val_rate,
            subset = 'training',
            seed = 2,
        )
        val_ds = image_dataset_from_directory(
            directory = train_directory,
            labels = 'inferred',
            label_mode = 'categorical',
            shuffle = True,
            batch_size = BATCH_SIZE,
            image_size = IMAGE_SIZE,
            validation_split = val_rate,
            subset = 'validation',
            seed = 2,
        )
    else:
        train_ds = image_dataset_from_directory(
            directory = train_directory,
            labels = 'inferred',
            label_mode = 'categorical',
            shuffle = True,
            batch_size = BATCH_SIZE,
            image_size = IMAGE_SIZE,
        )
        val_ds = image_dataset_from_directory(
            directory = val_directory,
            labels = 'inferred',
            label_mode = 'categorical',
            shuffle = True,
            batch_size = BATCH_SIZE,
            image_size = IMAGE_SIZE,
        )
    if test_directory == '':
        test_ds = val_ds
    else:
        test_ds = image_dataset_from_directory(
            directory = test_directory,
            labels = 'inferred',
            label_mode = 'categorical',
            shuffle = True,
            batch_size = BATCH_SIZE,
            image_size = IMAGE_SIZE,
        )
    return train_ds, val_ds, test_ds

def show_samples(dataset: Dataset, names: List[str]) -> None:
    fig, axes = plt.subplots(5, 5, figsize = (12, 12))
    axes = axes.flatten()
    
    sample_dataset = next(iter(dataset))
    for i in range(25):
        image = sample_dataset[0][i] / 255.0
        label = sample_dataset[1][i]
        label = np.argmax(label.numpy())
        axes[i].imshow(image)
        axes[i].set_title(names[label])
    
    plt.tight_layout(pad = 0.5)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Input the paths to the training, validation, and test data'
        )

    parser.add_argument(
        'train',
         type = str,
         help = 'Location of the training data'
         )
    parser.add_argument(
        '--val', 
        type = str, 
        default = '', 
        help = 'Location of validation data'
        )
    parser.add_argument(
        '--test', 
        type = str, 
        default = '',
        help = 'Location of test data'
        )
    parser.add_argument(
        '--val_percent', 
        type = float, 
        default = 0.2, 
        help = 'Percentation of train data used for val'
        )

    args = parser.parse_args()
    train_directory = args.train
    val_directory = args.val
    test_directory = args.test
    val_rate = args.val_percent
    #show_samples(dataset, names)

    train_ds, val_ds, test_ds = get_datasets(
                                    train_directory,
                                    val_directory,
                                    test_directory,
                                    val_rate
                                    )
    class_names = train_ds.class_names

    augment_layer = build_augment_layer(
                        horizontal = True,
                        vertical = False,
                        rotate = True,
                        zoom = True
                        )
    
    model = build_pretrained_model(shape = IMAGE_SIZE, 
                        class_number = len(class_names), 
                        augment_layer = augment_layer
                        )
    path_to_save = '/'
    train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size = AUTOTUNE)
    train_model(model = model, learning_rate = 10**-4, train_ds = train_ds, val_ds = val_ds, epochs = 20, path = path_to_save)
    evaluate_model(model, test_ds, class_names)
    save_model(model, path_to_save)
