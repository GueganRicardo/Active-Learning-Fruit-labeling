# Este programa:
# mostra num_images_to_display predictions realizadas
# calcula a accuracy com base em todas as imagens de teste

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 3:
    print("Usage: python new_active_learning.py <model_filename_keras> <num_images_to_display>")
    quit()

# Get command-line arguments
model_filename_keras = sys.argv[1]
num_images_to_display = int(sys.argv[2])


if os.path.exists(model_filename_keras):
    # Load existing model in Keras format
    model = keras.models.load_model(model_filename_keras)
    print("load do modelo:")
    print(model.summary())
else:
    print("MODEL NOT FOUND 404")
    quit()


test_data_dir = './test'

# Set the image size and batch size
img_width, img_height = 150, 150
batch_size = 32


test_datagen = ImageDataGenerator(rescale=1./255)


test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size)

# Evaluate the model on the test set
score = model.evaluate(test_generator)


# ... (previous code)

# Define a mapping for display labels
display_labels = {
    'freshapples': 'FA',
    'freshbanana': 'FB',
    'freshoranges': 'FO',
    'rottenapples': 'RA',
    'rottenbanana': 'RB',
    'rottenoranges': 'RO'
}

# Reset the test generator to the beginning
test_generator.reset()

for _ in range(num_images_to_display):
    # Get the next batch of images and labels
    test_data = next(test_generator)
    images, labels = test_data

    # Make predictions
    predictions = model.predict(images)

    # Get the class names
    class_names = list(test_generator.class_indices.keys())

    # Display the image and prediction probabilities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Display the image
    ax1.imshow(images[0])
    ax1.axis('off')
    ax1.set_title('Image')

    # Display the prediction probabilities as a bar plot with modified labels
    modified_labels = [display_labels[label] for label in class_names]
    ax2.bar(modified_labels, predictions[0])
    ax2.set_title('Prediction Probabilities')
    ax2.set_ylabel('Probability')
    ax2.set_ylim([0, 1])

    # Get the true label and convert it to a modified label
    true_label = class_names[np.argmax(labels[0])]
    modified_true_label = display_labels[true_label]

    # Get the predicted label and convert it to a modified label
    predicted_label = class_names[np.argmax(predictions[0])]
    modified_predicted_label = display_labels[predicted_label]

    # Display the true and predicted labels
    fig.suptitle(f'True: {modified_true_label}, Predicted: {modified_predicted_label}')
    
    # Show the plot
    plt.show()

print(f"\nTest Accuracy: {score[1]*100:.2f}%")
