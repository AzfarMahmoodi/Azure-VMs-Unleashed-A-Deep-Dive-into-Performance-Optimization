import os
import json
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import time

# Define the cluster configuration
cluster_spec = {
    "cluster": {
        "worker": ["10.0.0.4:12345", "10.0.0.5:12345", "10.0.0.6:12345"]
    },
    "task": {"type": "worker", "index": 0}  # Update the index for each worker
}
os.environ["TF_CONFIG"] = json.dumps(cluster_spec)

# Create the distributed strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Print cluster information
print("Cluster information:", strategy.cluster_resolver.cluster_spec())
print("Task type:", strategy.cluster_resolver.task_type)
print("Task id:", strategy.cluster_resolver.task_id)

# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Filter out classes 8 and 9
classes_to_remove = [8, 9]
train_mask = np.isin(y_train, classes_to_remove, invert=True).flatten()
test_mask = np.isin(y_test, classes_to_remove, invert=True).flatten()

x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test, y_test = x_test[test_mask], y_test[test_mask]

# Update class indices
y_train = np.array([y - 1 if y > 7 else y for y in y_train]).flatten()
y_test = np.array([y - 1 if y > 7 else y for y in y_test]).flatten()

# One-hot encoding
y_train = to_categorical(y_train, num_classes=8)
y_test = to_categorical(y_test, num_classes=8)

# Determine global and per-worker batch sizes
per_worker_batch_size = 64
num_workers = len(cluster_spec["cluster"]["worker"])
global_batch_size = per_worker_batch_size * num_workers

# Create training and testing datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(global_batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(global_batch_size)

# Display the number of images each worker will process
total_train_samples = len(x_train)
total_test_samples = len(x_test)
samples_per_worker_train = total_train_samples / num_workers
samples_per_worker_test = total_test_samples / num_workers

print(f"Total training samples: {total_train_samples}")
print(f"Total testing samples: {total_test_samples}")
print(f"Each worker will process approximately {samples_per_worker_train:.2f} training samples and {samples_per_worker_test:.2f} testing samples.")
# Define the model architecture
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001), input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(96, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(96, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        GlobalAveragePooling2D(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(8, activation='softmax')
    ])

    return model

model = build_model()

# Define learning rate schedule
def lr_schedule(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule(0))

# Train/test split
x_train_new, x_val, y_train_new, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Callbacks
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    mode='auto'
)

lr_scheduler = LearningRateScheduler(lr_schedule)

# Compile and train the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Start measuring time
start_time = time.time()

# Train the model
history = model.fit(
    x_train_new, 
    y_train_new, 
    epochs=5,
    validation_data=(x_val, y_val),
    shuffle=True, 
    callbacks=[learning_rate_reduction, lr_scheduler], 
    verbose=2
)

# Calculate time taken
end_time = time.time()
time_taken = end_time - start_time
print(f"Time taken: {time_taken:.5f} seconds")

# Evaluate the model
accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy[1]*100:.2f}%")



# Save the model
model.save("modelqwe.keras")
