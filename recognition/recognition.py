import numpy as np
import os

import tensorflow as tf
from keras import layers
from tensorflow.keras.models import Sequential


input_shape = (12, 157)

def load_data(data_dir):
    file_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    class_indices = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for f in os.listdir(class_dir):
            file_paths.append(os.path.join(class_dir, f))
            labels.append(class_indices[class_name])

    return file_paths, labels


model = Sequential()
model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
model.add(layers.MaxPooling1D(pool_size=2))
# model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
# model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'])

# print(model.summary())

data_dir = "data_set"

data_type_train = "train"
data_train = os.path.join(data_dir, data_type_train)

data_type_val = "val"
data_val = os.path.join(data_dir, data_type_val)

data_type_test = "test"
data_test = os.path.join(data_dir, data_type_test)

train_file_paths, train_labels = load_data(data_train)
val_file_paths, val_labels = load_data(data_val)
test_file_paths, test_labels = load_data(data_test)

arrays1 = [np.load(file) for file in train_file_paths]
arrays2 = [np.load(file) for file in val_file_paths]
arrays3 = [np.load(file) for file in test_file_paths]
train_set = np.stack(arrays1)
val_set = np.stack(arrays2)
test_set = np.stack(arrays3)

train_dataset = tf.data.Dataset.from_tensor_slices((train_set, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((val_set, val_labels))
val_dataset = val_dataset.shuffle(buffer_size=1024).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_set, test_labels))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(32)

model.fit(train_dataset, validation_data=val_dataset, epochs=10)

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

