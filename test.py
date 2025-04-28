# Testing Script: test_unet.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU

# Load trained model
model = tf.keras.models.load_model("unet_model.h5")

# Load test data (Ensure same preprocessing as training script)
def load_test_data():
    test_image_dir = "./Task08_HepaticVessel/imagesTr/"
    test_mask_dir = "./Task08_HepaticVessel/labelsTr/"
    
    image_files = sorted([f for f in os.listdir(test_image_dir) if f.endswith(".nii.gz")])
    mask_files = sorted([f for f in os.listdir(test_mask_dir) if f.endswith(".nii.gz")])
    
    data = []
    labels = []
    
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(test_image_dir, img_file)
        mask_path = os.path.join(test_mask_dir, mask_file)
        
        img_3d = load_nifti(img_path)
        mask_3d = load_nifti(mask_path)
        
        slices = slice_3d_to_2d(img_3d, mask_3d)
        for img_slice, mask_slice in slices:
            data.append(img_slice)
            labels.append(mask_slice)
    
    data = np.array(data).astype(np.float32)
    labels = np.array(labels).astype(np.float32)
    
    data = data / np.max(data)  # Normalize
    data = np.expand_dims(data, axis=-1)
    labels = np.expand_dims(labels, axis=-1)
    
    return data, labels

X_test, y_test = load_test_data()

# Evaluation Metrics
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(np.float32)
    
    iou = MeanIoU(num_classes=2)(y_test, y_pred).numpy()
    dice = dice_coefficient(y_test, y_pred)
    precision = np.sum(y_pred * y_test) / (np.sum(y_pred) + 1e-7)
    
    print(f"IoU: {iou:.4f}, Dice: {dice:.4f}, Precision: {precision:.4f}")

evaluate_model(model, X_test, y_test)
