import os
import glob
import cv2
import numpy as np
from keras.models import Model # type: ignore
from keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Flatten, Dense, Reshape, Concatenate # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.layers import MaxPooling2D # type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # For visualization


def load_and_preprocess(data_dir, new_size=32):
    """Loads and preprocesses image data."""
    X_train = []
    y_train = []
    image_class = {'Boot': 0, 'Sandal': 1, 'Shoe': 2}
    for folder in os.listdir(data_dir):
        if folder in image_class:  
            files = glob.glob(os.path.join(data_dir, folder, '*.jpg'))
            for file in files:
                try:
                    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (new_size, new_size))
                    img = img / 255.0
                    X_train.append(img)
                    y_train.append(image_class[folder])
                except Exception as e:
                    print(f"Error processing {file}: {e}")  

    return np.array(X_train), np.array(y_train)


def create_model(img_shape):
    """Creates a simple CNN model."""
    input_layer = Input(shape=img_shape)
    
  
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pool = MaxPooling2D((2, 2))(conv2) 
    flatten = Flatten()(pool)
    dense1 = Dense(128, activation='relu')(flatten)
    output_layer = Dense(3, activation='softmax')(dense1)  

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

train_dir = 'C:/Users/leewa/OneDrive/Desktop/CNN/Footwear'
img_size = 32

X_train, y_train = load_and_preprocess(train_dir, img_size)

if X_train.size ==0:
    print("No valid images found. Check your data directory and file extensions")
    exit() 

img_shape = (img_size, img_size, 1)  
X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 1)  

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = create_model(img_shape)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))




plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()