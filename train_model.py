import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.image as mpimg

print('Setting Up...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define constants and parameters
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.2
AUGMENTATION_RATE = 0.5 

# Paths and file names
DATA_DIR = '.'
CSV_FILE = os.path.join(DATA_DIR, 'driving_log.csv')
IMG_DIR = os.path.join(DATA_DIR, 'IMG')

def load_data():
    """Load data from CSV file"""
    print('Loading data...')
    data = pd.read_csv(CSV_FILE)
    
    # Extract center image paths and steering angles
    image_paths = data.iloc[:, 0].values  # First column (center camera)
    steering_angles = data.iloc[:, 3].values  # Fourth column (steering)
    
    # Clean up the image paths (in case)
    image_paths = [path.strip() for path in image_paths]
    
    image_paths = [os.path.join(IMG_DIR, os.path.basename(path)) for path in image_paths]
    
    print(f'Total images: {len(image_paths)}')
    return image_paths, steering_angles

def preprocess_image(img):
    # Crop the road area
    img = img[60:135, :, :]
    
    # Convert to YUV color space
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    # Apply Gaussian Blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Resize to 200x66 pixels
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize
    img = img / 255.0
    
    return img

# Data Augmentation
def random_augment(img, steering):
    # Make a copy of the image and steering angle
    aug_img = img.copy()
    aug_steering = steering
    
    # 1. Random flip (50% chance)
    if random.random() > 0.5:
        aug_img = cv2.flip(aug_img, 1)  
        aug_steering = -aug_steering 
    
    # 2. Random brightness 
    if random.random() > 0.5:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_RGB2HSV)
        ratio = 0.25 + random.random() * 0.5  
        hsv[:,:,2] = hsv[:,:,2] * ratio
        aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 3. Random zoom 
    if random.random() > 0.5:
        zoom = random.uniform(0.8, 1.0) 
        h, w = aug_img.shape[:2]
        h_crop = int(h * zoom)
        w_crop = int(w * zoom)
        h_start = random.randint(0, h - h_crop)
        w_start = random.randint(0, w - w_crop)
        aug_img = aug_img[h_start:h_start+h_crop, w_start:w_start+w_crop]
        aug_img = cv2.resize(aug_img, (w, h))
    
    # 4. Random pan (shift image)
    if random.random() > 0.5:
        h, w = aug_img.shape[:2]
        tx = random.randint(-w//10, w//10)  
        ty = random.randint(-h//10, h//10) 
        transform_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        aug_img = cv2.warpAffine(aug_img, transform_matrix, (w, h))
    
    # 5. Random rotation
    if random.random() > 0.5:
        h, w = aug_img.shape[:2]
        rotation = random.uniform(-10, 10) 
        rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1)
        aug_img = cv2.warpAffine(aug_img, rotation_matrix, (w, h))
    
    return aug_img, aug_steering

def batch_generator(image_paths, steering_angles, batch_size=32, augment=True):
    while True:
        # Create batch arrays
        batch_imgs = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        batch_steering = np.zeros(batch_size)
        
        # Generate random batch indices
        indices = np.random.choice(len(image_paths), batch_size)
        
        for i, idx in enumerate(indices):
            img_path = image_paths[idx]
            img = mpimg.imread(img_path)
            steering = steering_angles[idx]
            
            # Decide whether to augment this image
            if augment and random.random() < AUGMENTATION_RATE:
                img, steering = random_augment(img, steering)
            
            # Preprocess image
            img = preprocess_image(img)
            
            # Add to batch
            batch_imgs[i] = img
            batch_steering[i] = steering
        
        yield batch_imgs, batch_steering

def create_nvidia_model():
    model = Sequential([
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=INPUT_SHAPE),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(1164, activation='relu'),
        Dropout(0.2),
        Dense(100, activation='relu'),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='relu'),
        Dense(1) 
    ])
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))
    return model

# Plot the training and validation loss"
def plot_training_history(history):
    plt.figure(figsize=(10, 8))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_history.png')
    plt.show()

# Analyze and plot the distribution of steering angles
def analyze_steering_distribution(steering_angles):
    plt.figure(figsize=(10, 6))
    plt.hist(steering_angles, bins=50)
    plt.title('Steering Angle Distribution')
    plt.xlabel('Steering Angle')
    plt.ylabel('Frequency')
    plt.savefig('steering_distribution.png')
    plt.show()

def main():
    # Load data
    image_paths, steering_angles = load_data()
    
    # Analyze steering distribution
    analyze_steering_distribution(steering_angles)
    
    # Split data into training and validation 
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, 
        steering_angles, 
        test_size=VALIDATION_SPLIT, 
        random_state=42
    )
    
    print(f'Training samples: {len(X_train)}, Validation samples: {len(X_val)}')
    
    # Create generators
    train_generator = batch_generator(X_train, y_train, batch_size=BATCH_SIZE, augment=True)
    val_generator = batch_generator(X_val, y_val, batch_size=BATCH_SIZE, augment=False)
    
    #steps per epoch
    steps_per_epoch = len(X_train) // BATCH_SIZE
    validation_steps = len(X_val) // BATCH_SIZE
    
    # train model
    model = create_nvidia_model()
    print(model.summary())
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        verbose=1
    )
    
    # Plot training 
    plot_training_history(history)
    
    # Save model
    model.save('model.h5')
    print("Model saved as 'model.h5'")

if __name__ == "__main__":
    main()