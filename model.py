import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Input, Flatten, Conv2D, SpatialDropout2D, Activation, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3
from tensorflow.python.data import Dataset
from typing import Tuple, List
import matplotlib.pyplot as plt

def build_augment_layer(horizontal: bool, vertical: bool, rotate: bool, zoom: bool):
    augment_layer = tf.keras.Sequential()

    if horizontal:
        augment_layer.add(tf.keras.layers.RandomFlip('horizontal'))
    if vertical:
        augment_layer.add(tf.keras.layers.RandomFlip('vertical'))
    if rotate:
        augment_layer.add(tf.keras.layers.RandomRotation(0.2))
    if zoom:
        augment_layer.add(tf.keras.layers.RandomZoom(0.2))

    return augment_layer

def build_pretrained_model(shape: Tuple[int], class_number: int, augment_layer = None):
    inp = Input(shape = (*shape, 3))
    x = inp
    if augment_layer:
        x = augment_layer(x)
    base_model = InceptionV3(
        input_tensor = x, 
        include_top = False, 
        weights = 'imagenet'
        )
    head_model = base_model.output
    head_model = Flatten()(head_model)
    head_model = BatchNormalization()(head_model)
    head_model = Dropout(rate = 0.75)(head_model)
    head_model = Dense(units = 256, activation = 'relu')(head_model)
    head_model = Dropout(rate = 0.75)(head_model)
    head_model = Dense(units = class_number, activation = 'softmax')(head_model)

    return Model(inputs = base_model.input, outputs = head_model)

def conv_blocks(inp, filters: List[int], kernel_sizes: List[int], rate = 0.75):
    x = inp
    for fil, ker in zip(filters, kernel_sizes):
        if ker == 2:
            x = MaxPool2D()(x)
            x = Conv2D(fil, kernel_size = 3, padding = 'same')(x)
        else:
            x = Conv2D(fil, ker, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(rate)(x)
    return x

def build_model(shape: Tuple[int], class_number: int, augment_layer = None):
    inp = Input(shape = (*shape, 3))
    x = inp
    if augment_layer:
        x = augment_layer(x)
    filters = [32, 64, 64, 128, 128]
    kernel_sizes = [3, 2, 3, 2, 3]
    x = conv_blocks(inp, filters, kernel_sizes, rate = 0.25)
    x = Flatten()(x)
    x = Dense(units = 256, activation = 'relu')(x)
    x = Dense(units = class_number, activation = 'softmax')(x)
    return Model(inputs = inp, outputs = x)

    
def train_model(model, learning_rate, train_ds, val_ds, epochs, path):
    
    model.compile(optimizer = Adam(learning_rate = learning_rate), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    early_stopping = EarlyStopping(monitor = 'val_categorical_accuracy', patience = 20, restore_best_weights= True)
    model_checkpoint = ModelCheckpoint(filepath = path, monitor = 'val_categorical_accuracy', save_weights_only = True, save_best_only= True)
    hist = model.fit(train_ds, validation_data = val_ds, epochs = epochs, callbacks = [early_stopping, model_checkpoint])
    return hist

def evaluate_model(model: Model, test_ds: Dataset, class_names: List[str]):
    predictions = model.predict(test_ds)
    prediction_labels = tf.argmax(predictions, axis = 1)
    true_labels = tf.concat([tf.argmax(y, axis = 1) for x, y in test_ds], axis = 0)
    cm = confusion_matrix(true_labels, prediction_labels)
    cmd = ConfusionMatrixDisplay(cm, display_labels = class_names)
    cmd.plot()
    plt.xticks(rotation = 90)
    plt.savefig('confusionmatrix.jpg')

def load_model_weights(model, path_to_weights):
    model.load_weights(path_to_weights)
    print('Model weights loaded')

def save_model(model, path):
    model.save(path)
    print('Model saved')

def load_model(path_to_model):
    model = tf.keras.models.load_model(path_to_model)
    return model