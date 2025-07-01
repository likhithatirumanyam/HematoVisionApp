from google.colab import files

files.upload()
import os

os.makedirs("/root/.kaggle", exist_ok=True)

!mv kaggle.json /root/.kaggle/

!chmod 600 /root/.kaggle/kaggle.json
!pip install -q Kaggle
!kaggle datasets download -d paultimothymooney/blood-cells
import zipfile

# Unzip the blood cells dataset

with zipfile.ZipFile("blood-cells.zip", 'r') as zip_ref:

  zip_ref.extractall("blood_cells_dataset")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import os
import random

# Path to training images
train_path = "blood_cells_dataset/dataset2-master/dataset2-master/images/TRAIN"
class_names = os.listdir(train_path)

# Set image size
img_size = (128, 128)

# Create a 2x2 plot (for 4 classes)
plt.figure(figsize=(10, 8))

for i, class_name in enumerate(class_names):
    class_folder = os.path.join(train_path, class_name)
    image_name = random.choice(os.listdir(class_folder))
    image_path = os.path.join(class_folder, image_name)

    # Read and resize image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)

    # Show image
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.title(class_name)
    plt.axis('off')

plt.suptitle("Random Images from Each Blood Cell Class", fontsize=16)
plt.tight_layout()
plt.show()
import os
import shutil
import random

# ✅ Corrected paths
original_train_path = "blood_cells_dataset/dataset2-master/dataset2-master/images/TRAIN"
original_test_path = "blood_cells_dataset/dataset2-master/dataset2-master/images/TEST"
base_output_path = "blood_cells_split"

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Set seed for reproducibility
random.seed(42)

# Create split folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(base_output_path, split), exist_ok=True)

# Merge TRAIN and TEST folders for each class
all_classes = os.listdir(original_train_path)

for class_name in all_classes:
    # Collect all image paths from TRAIN and TEST
    train_class_dir = os.path.join(original_train_path, class_name)
    test_class_dir = os.path.join(original_test_path, class_name)

    images = []
    if os.path.isdir(train_class_dir):
        images += [os.path.join(train_class_dir, f) for f in os.listdir(train_class_dir)]
    if os.path.isdir(test_class_dir):
        images += [os.path.join(test_class_dir, f) for f in os.listdir(test_class_dir)]

    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in splits.items():
        split_class_dir = os.path.join(base_output_path, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img_path in split_images:
            img_name = os.path.basename(img_path)
            dst = os.path.join(split_class_dir, img_name)
            shutil.copyfile(img_path, dst)

print("✅ Blood cell images successfully split into train, val, and test sets.")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,                # Normalize pixel values
    rotation_range=20,             # Rotate images by up to 20 degrees
    width_shift_range=0.2,         # Shift image left/right
    height_shift_range=0.2,        # Shift image up/down
    zoom_range=0.2,                # Zoom in/out
    horizontal_flip=True,          # Flip image horizontally
    fill_mode='nearest'            # Fill in any empty pixels
)

# Validation and test sets should not be augmented
val_test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'blood_cells_split/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    'blood_cells_split/val',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    'blood_cells_split/test',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Don't shuffle test data so evaluation is consistent
)
# Preview some augmented images
augmented_images, _ = next(train_generator)

plt.figure(figsize=(10, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(augmented_images[i])
    plt.axis('off')
plt.suptitle("Augmented Training Images")
plt.tight_layout()
plt.show()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ✅ Replace this with dynamic code if train_generator is already defined
# num_classes = len(train_generator.class_indices)
num_classes = 4  # or 5, depending on your dataset structure

# Build CNN model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
import zipfile, os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ✅ UNZIP DATASET
zip_path = '/content/blood-cells.zip'  # Make sure you uploaded this
extract_path = '/content/'  # Always extract to root in Colab

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# ✅ AUTO-DETECT main dataset folder
def find_dataset_folder(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if set(['train', 'val', 'test']).issubset(set(dirs)):
            return root
    raise FileNotFoundError("Could not find 'train', 'val', and 'test' folders inside extracted zip.")

base_dir = find_dataset_folder(extract_path)

train_path = os.path.join(base_dir, 'train')
val_path = os.path.join(base_dir, 'val')
test_path = os.path.join(base_dir, 'test')

# ✅ Image and batch settings
image_size = (224, 224)
batch_size = 32

# ✅ Count classes automatically
num_classes = len(os.listdir(train_path))

# ✅ Image Generators
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, shear_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_datagen.flow_from_directory(
    test_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ✅ Load MobileNetV2 base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# ✅ Custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ✅ Compile
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    verbose=1
)

# ✅ Evaluate
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Get true labels and predictions
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Get class labels
class_labels = list(test_generator.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))
model.save("Blood_Cell.h5")
from google.colab import files
files.download("Blood_Cell.h5")








