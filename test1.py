import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from skimage.transform import resize

# ===============================
# Configuration
# ===============================
DATA_DIR = "./Task08_HepaticVessel"  # Path to dataset
LOG_DIR = "./logs"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 2
EPOCHS = 50
INPUT_SHAPE = (128, 128, 64, 1)  # Resize if needed
NUM_CLASSES = 3

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===============================
# Data Loader
# ===============================
def load_nifti_file(filepath):
    img = nib.load(filepath).get_fdata()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize
    img = resize(img, INPUT_SHAPE[:3], anti_aliasing=True)
    img = np.expand_dims(img, axis=-1)
    return img

def data_generator(dataset_json, mode="train"):
    import json
    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
    if mode == "train":
        data = dataset['training']
    else:
        data = dataset['test']

    for item in data:
        if mode == "train":
            img_path = os.path.join(DATA_DIR, item['image'])
            mask_path = os.path.join(DATA_DIR, item['label'])
            img = load_nifti_file(img_path)
            mask = load_nifti_file(mask_path)
            mask = tf.one_hot(tf.cast(mask[..., 0], tf.int32), NUM_CLASSES)
            yield img, mask
        else:
            img_path = os.path.join(DATA_DIR, item)
            img = load_nifti_file(img_path)
            yield img

def build_dataset(dataset_json, mode="train"):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(dataset_json, mode),
        output_signature=(
            tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=INPUT_SHAPE[:3] + (NUM_CLASSES,), dtype=tf.float32),
        ) if mode == "train" else tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32)
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ===============================
# Build U-Net Model
# ===============================
def build_unet():
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D(2)(x)
    x = layers.Conv3D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(2)(x)
    x = layers.Conv3D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv3DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv3DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv3D(NUM_CLASSES, 1, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ===============================
# Train
# ===============================
dataset_json = os.path.join(DATA_DIR, 'dataset.json')
train_dataset = build_dataset(dataset_json, mode="train")
model = build_unet()
model.summary()

callbacks_list = [
    callbacks.ModelCheckpoint(f'{CHECKPOINT_DIR}/model.h5', save_best_only=True),
    callbacks.TensorBoard(LOG_DIR),
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
]

history = model.fit(train_dataset, validation_data=train_dataset, epochs=EPOCHS, callbacks=callbacks_list)

# ===============================
# Test
# ===============================
test_dataset = build_dataset(dataset_json, mode="test")
predictions = model.predict(test_dataset)
np.save('predictions.npy', predictions)