import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import shutil
from scipy.stats import entropy


expert_data_dir = './expert_labelled'
machine_data_dir = './machine_labelled'
all_data_dir = './all_labelled'
img_width, img_height = 150, 150
batch_size = 32

assuranceValue = 0.9 #how certain the model needs to be to accept
nEpochsFirst = 10 #for the pretrain data
nEpochsOthers = 3 #for each next iteration
QueryStartegy = "lc" #options: lc sm eb


class ImageLabeler:
    def __init__(self, root, image_dir, button_names):
        self.query_strat = QueryStartegy
        self.root = root
        self.image_dir = image_dir
        self.button_names = button_names
        self.current_image_index = 0
        self.labeled_data = []
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print("Estão "+ str(len(self.image_files)) + " imagens na pasta CROP")
        self.image_files = self.image_files[:100]
        lista_certas, lista_incertas = self.more_assur_images(self.image_files)
        self.image_files = lista_incertas
        self.image_label = tk.Label(root)
        self.image_label.pack()
        self.buttons = []
        for name in button_names:
            button = tk.Button(root, text=name, command=lambda n=name: self.on_button_click(n))
            button.pack(side=tk.LEFT)
            self.buttons.append(button)
        self.load_image()

    def load_image(self):
        if self.current_image_index < len(self.image_files):
            image_path = os.path.join(self.image_dir, self.image_files[self.current_image_index])
            image = Image.open(image_path)
            image = image.resize((150, 150))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        else:
            self.root.destroy()  
            self.create_or_load_neural_network(nEpochsOthers)
            

    def on_button_click(self, button_name):
        #print(f"Button '{button_name}' clicked for image '{self.image_files[self.current_image_index]}'")
        self.labeled_data.append((self.image_files[self.current_image_index], button_name))
        self.current_image_index += 1
        self.load_image()

    def create_or_load_neural_network(self, n_epochs):
        if os.path.exists(model_filename_keras):
            model = keras.models.load_model(model_filename_keras)
            print("Model loaded.")
        else:
            model = self.create_neural_network()
            print("Model created.")
        data = self.labeled_data
        images = [np.array(Image.open(os.path.join(self.image_dir, img)).resize((150, 150))) for img, _ in data]
        for i in range(len(images)):
            if images[i].shape[2] == 4:
                images[i] = images[i][:, :, 0:3]
        all_labelled_dir = "all_labelled"
        if not os.path.exists(all_labelled_dir):
            print("###################################")
            print("ERROR - missing all_labelled dir")
            print("###################################")
            quit()
        machine_labelled_dir = "machine_labelled"
        if not os.path.exists(machine_labelled_dir):
            os.makedirs(machine_labelled_dir)
            for button_name in self.button_names:
                os.makedirs(os.path.join(machine_labelled_dir, button_name))
        expert_labelled_dir = "expert_labelled"
        if not os.path.exists(expert_labelled_dir):
            os.makedirs(expert_labelled_dir)
            for button_name in self.button_names:
                os.makedirs(os.path.join(expert_labelled_dir, button_name))
        for img, label in data:
            ex_label_dir = os.path.join(expert_labelled_dir, label)
            all_label_dir = os.path.join(all_labelled_dir, label)
            img_path = os.path.join(self.image_dir, img)
            ex_dest_path = os.path.join(ex_label_dir, img)
            all_dest_path = os.path.join(all_label_dir, img)
            shutil.copy(img_path, ex_dest_path)
            shutil.copy(img_path, all_dest_path)
            os.remove(img_path)
        train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
        train_generator = train_datagen.flow_from_directory(all_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode="categorical",
                                                    shuffle=True)
        model.fit(train_generator, epochs=n_epochs)
        model.save("image_classification_model.keras")
        print("Model saved.")

    def load_images_from_directory(self, directory):
        images = []
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                for img_file in os.listdir(subdir_path):
                    img_path = os.path.join(subdir_path, img_file)
                    img = np.array(Image.open(img_path).resize((150, 150)))
                    if img.shape[2] == 4:
                        img = img[:, :, 0:3]
                    images.append(img)
        return images

    def load_labels_from_directory(self, directory):
        labels = []
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                labels.extend([self.button_names.index(subdir)] * len(os.listdir(subdir_path)))
        return labels

    def create_neural_network(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(150,150, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(6, activation='softmax'))  
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def predict(self, model, processed_image):
        processed_image_array = np.array(processed_image.resize((150, 150)))
        if processed_image_array.shape[2] == 4:
            processed_image_array = processed_image_array[:,:,0:3]
        processed_image_array = processed_image_array / 255.0
        processed_image_array = processed_image_array.reshape(1, 150, 150, 3)
        predictions = model.predict(processed_image_array)
        return predictions[0]

    def visualize_predictions(self, test_image, predictions, class_names):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(test_image)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.barh(class_names, predictions)
        plt.xlabel('Probability')
        plt.title('Class Probabilities')
        plt.show()

    def more_assur_images(self, list_images):
        predictions = list()
        if os.path.exists(model_filename_keras):
            model = keras.models.load_model(model_filename_keras)
            for test_image_file in list_images:
                test_image_path = os.path.join(image_dir, test_image_file)
                test_image = Image.open(test_image_path)
                test_image = test_image.resize((150, 150))
                predictions.append(self.predict(model, test_image)) 
            mais_assurances, menos_assurances = self.apply_query_strategy(predictions, list_images)
        else: # não há model
            print("Error: You need to restart with a 1sttime.txt file...")
            quit()
        nomes_mais_assurances = list()
        nomes_menos_assurances = list()
        for i in mais_assurances:
            nomes_mais_assurances.append(list_images[i])
        for i in menos_assurances:
            nomes_menos_assurances.append(list_images[i])
        for i in mais_assurances:
            nome_certa = list_images[i]
            img_path = os.path.join(self.image_dir, nome_certa)
            index_max = np.argmax(predictions[i])
            class_name = self.button_names[index_max]
            ma_dest_path = os.path.join("machine_labelled",class_name,nome_certa)
            all_dest_path = os.path.join("all_labelled",class_name,nome_certa)
            shutil.copy(img_path, ma_dest_path)
            shutil.copy(img_path, all_dest_path)
            os.remove(img_path)
        print("Certain: ",len(nomes_mais_assurances))
        print("Uncertain: ",len(nomes_menos_assurances))
        return nomes_mais_assurances, nomes_menos_assurances

    def apply_query_strategy(self, predictions, list_images):
        mais_assurances = list()
        menos_assurances = list()
        if self.query_strat == "lc":
            maxs = [max(pred) for pred in predictions]
            sorted_indices = sorted(enumerate(maxs), key=lambda x: x[1])
            menos_assurances = [index for index, value in sorted_indices if value < assuranceValue]
            mais_assurances = [index for index, value in sorted_indices if value >= assuranceValue]
            if len(menos_assurances) > 20:
                menos_assurances = menos_assurances[:20]
        elif self.query_strat == "sm":
            smallest_margins = []
            for pred in predictions:
                sorted_indices = np.argsort(pred)[::-1]  # Sort indices in descending order
                margin = pred[sorted_indices[0]] - pred[sorted_indices[1]]
                smallest_margins.append(margin)
            sorted_indices_by_margin = sorted(enumerate(smallest_margins), key=lambda x: x[1])
            menos_assurances = [index for index, value in sorted_indices_by_margin if value < 0.99]
            mais_assurances = [index for index, value in sorted_indices_by_margin if value >= 0.99]
            if len(menos_assurances) > 20:
                menos_assurances = menos_assurances[:20]
        elif self.query_strat == "eb":
            entropies = [entropy(pred) for pred in predictions]
            entropies = np.argsort(entropies)[::-1]
            sorted_indices_by_entropy = sorted(enumerate(entropies), key=lambda x: x[1])
            menos_assurances = [index for index, value in sorted_indices_by_entropy if value > 0.0895]
            mais_assurances = [index for index, value in sorted_indices_by_entropy if value <= 0.0895]
            if len(menos_assurances) > 20:
                menos_assurances = menos_assurances[:20]
        return mais_assurances, menos_assurances
            


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Labeling Tool")
    model_filename_keras = 'image_classification_model.keras'
    image_dir = "./yolov4-custom-functions/detections/crop"
    if image_dir:
        button_names = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]
        labeler = ImageLabeler(root, image_dir, button_names)
        print("Not First Time")
        root.mainloop()

