#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import mediapipe as mp
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# In[2]:


# Function to load images from a folder
def load_images_from_folder(folder, target_size=(50, 50)):
    images = []
    labels = []
    for alphabet in os.listdir(folder):
        alphabet_folder = os.path.join(folder, alphabet)
        if os.path.isdir(alphabet_folder):
            print(f"Loading images from training folder: {alphabet_folder}")
            for filename in os.listdir(alphabet_folder):
                if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                    img_path = os.path.join(alphabet_folder, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, target_size)  # Resize image to target size
                        images.append(img)
                        labels.append(alphabet)  # Combine alphabet and letter as label
                    else:
                        print(f"Ignoring non-image file: {filename}")
    print("Number of images loaded:", len(images))
    print("Number of labels loaded:", len(labels))
    return images, labels

# Function to preprocess images with Gaussian blur and binary thresholding
def preprocess_images(images):
    preprocessed_images = []
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
    
    for image in images:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply binary thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Resize image to desired dimensions
        resized = cv2.resize(binary, (50, 50))
        
        # Extract hand landmarks using MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb_image)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]  # Assuming only one hand is present
            # Process landmarks as needed
            # For example, you can extract coordinates of specific landmarks
            # and use them as features for your model
        else:
            pass  # If no hand landmarks are detected
        
        # Append preprocessed image to list
        preprocessed_images.append(resized)
    
    return np.array(preprocessed_images)

# Load images from train and test folders
train_images, train_labels = load_images_from_folder(r"E:\archive\asl_alphabet_train\asl_alphabet_train")
test_images, test_labels = load_images_from_folder(r"E:\archive\asl_alphabet_test")
# Preprocess train and test images
X_train_preprocessed = preprocess_images(train_images)
X_test_preprocessed = preprocess_images(test_images)


# In[ ]:


# Perform train-test split
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_preprocessed, train_labels, test_size=0.2, random_state=42)

# Convert labels to numerical format using label encoding
label_encoder = LabelEncoder()
all_labels = np.concatenate((y_train_final, y_val, test_labels))  # Combine all labels
label_encoder.fit(all_labels)  # Fit the encoder on all labels

# Transform train, validation, and test labels
y_train_encoded = label_encoder.transform(y_train_final)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(test_labels)

# Define the batch size for training
batch_size = 32

# Create an ImageDataGenerator instance for data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by up to 20%
    height_shift_range=0.2,  # Randomly shift images vertically by up to 20%
    shear_range=0.2,  # Shear intensity (shear angle in counter-clockwise direction)
    zoom_range=0.2,  # Randomly zoom images by up to 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Strategy for filling in newly created pixels
)

import tensorflow as tf

# Define a function to return the generator
def train_data_generator():
    while True:
        for batch in datagen.flow(
            np.array(X_train_final),  # Input images
            y_train_encoded,  # Labels
            batch_size=batch_size,  # Batch size
            shuffle=True  # Shuffle the data
        ):
            yield (batch[0], batch[1])

# Define a function to return the validation generator
def val_data_generator():
    while True:
        for batch in datagen.flow(
            np.array(X_val),  # Input images
            y_val_encoded,  # Labels
            batch_size=batch_size,  # Batch size
            shuffle=False  # Do not shuffle the data
        ):
            yield (batch[0], batch[1])

# Define a function to return the test generator
def test_data_generator():
    while True:
        for batch in datagen.flow(
            np.array(X_test_preprocessed),  # Input images
            y_test_encoded,  # Labels
            batch_size=batch_size,  # Batch size
            shuffle=False  # Do not shuffle the data
        ):
            yield (batch[0], batch[1])

from tensorflow.keras.layers import Input

# Define the CNN architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(96, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(27, activation='softmax')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data_generator(),
    epochs=20,  # Increase the number of epochs
    steps_per_epoch=len(X_train_final) // batch_size,
    validation_data=val_data_generator(),
    validation_steps=len(X_val) // batch_size
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data_generator(), steps=len(X_test_preprocessed) // batch_size)
print("Test accuracy:", test_accuracy)


# In[ ]:





# In[ ]:




