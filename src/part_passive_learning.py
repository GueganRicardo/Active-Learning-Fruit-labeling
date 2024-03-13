import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set the path to your data
train_data_dir = './train'

# Set the image size and batch size
img_width, img_height = 150, 150
batch_size = 32

# Set the number of files to use for each class
n_files = 50
incremental = False

# Create data generator for training
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Function to get limited files from each class
def get_limited_files(directory, n_files):
    class_count = {}
    limited_files = []

    for root, dirs, files in os.walk(directory):
        for class_name in dirs:
            class_count[class_name] = 0
            class_path = os.path.join(root, class_name)

            for file in os.listdir(class_path):
                if class_count[class_name] >= n_files:
                    break

                file_path = os.path.join(class_path, file)
                limited_files.append(file_path)
                class_count[class_name] += 1

    return limited_files, class_count

train_files, train_class_count = get_limited_files(train_data_dir, n_files)

print("Number of files per class used for training:", train_class_count)

# Calculate steps per epoch based on the desired number of files
steps_per_epoch = len(train_files) // batch_size

# Create data generator with limited files
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode="categorical",
                                                    shuffle=True)

# Manually limit the number of training samples
train_generator.samples = n_files * len(train_class_count)

# Build the CNN model
if incremental:
    model = keras.saving.load_model("passive_model.keras")
else:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(6, activation='softmax'))  
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, steps_per_epoch=steps_per_epoch)

# Save the model with the specified suffix
model.save(f"passive_model.keras")
print("Model saved successfully.")
