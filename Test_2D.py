import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import random
from tensorflow import keras

# Configurations
IMAGE_PATH = '2D_Sliced_Images/'
MASK_PATH = '2D_Sliced_Masks/'
IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 8
NUM_CLASSES = 3

# Load file paths
image_files = sorted([os.path.join(IMAGE_PATH, f) for f in os.listdir(IMAGE_PATH) if f.endswith('.png')])
mask_files = sorted([os.path.join(MASK_PATH, f) for f in os.listdir(MASK_PATH) if f.endswith('.png')])

# Data Generator
def data_generator(image_list, mask_list):
    for img_path, mask_path in zip(image_list, mask_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.digitize(mask, bins=[85, 170])
        img = np.expand_dims(img, axis=-1)
        yield img, tf.keras.utils.to_categorical(mask, NUM_CLASSES)

def create_dataset(image_list, mask_list):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(image_list, mask_list),
        output_signature=(
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES), dtype=tf.float32)
        )
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = create_dataset(image_files, mask_files)
model = keras.models.load_model(r"D:\python\Hapetic2\checkpoints\model.h5", custom_objects={"dice_loss": None})

# Dice Loss
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / (denominator + 1e-6)

# Per-class Dice
def dice_per_class(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    dice_scores = []
    for i in range(NUM_CLASSES):
        intersection = tf.reduce_sum(y_true[..., i] * y_pred[..., i])
        denominator = tf.reduce_sum(y_true[..., i] + y_pred[..., i])
        dice = (2. * intersection) / (denominator + 1e-6)
        dice_scores.append(dice.numpy())
    return dice_scores

# Per-class IoU
def iou_per_class(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    iou_scores = []
    for i in range(NUM_CLASSES):
        intersection = tf.reduce_sum(tf.cast((y_pred == i) & (y_true == i), tf.float32))
        union = tf.reduce_sum(tf.cast((y_pred == i) | (y_true == i), tf.float32))
        iou = intersection / (union + 1e-6)
        iou_scores.append(iou.numpy())
    return iou_scores

# Visualize Predictions
def visualize_predictions(img, mask, pred_mask, title="Prediction Overlay"):
    img = np.squeeze(img, axis=-1)
    true_mask = np.argmax(mask, axis=-1)
    pred_mask = np.argmax(pred_mask, axis=-1)

    overlay = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    overlay[true_mask == 1] = [255, 0, 0]
    overlay[pred_mask == 1] = [0, 255, 0]
    overlay[(true_mask == 1) & (pred_mask == 1)] = [0, 0, 255]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("CT Scan")

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, cmap='jet', alpha=0.6)
    plt.title("Ground Truth Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap='gray')
    plt.imshow(overlay, alpha=0.5)
    plt.title(title)
    plt.show()

# Test on random image
def test_random_image():
    idx = random.randint(0, len(image_files) - 1)
    img = cv2.imread(image_files[idx], cv2.IMREAD_GRAYSCALE) / 255.0
    mask = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE)
    mask = np.digitize(mask, bins=[85, 170])
    img = np.expand_dims(img, axis=(0, -1))
    mask = tf.keras.utils.to_categorical(mask, NUM_CLASSES)
    pred_mask = model.predict(img)

    iou = iou_per_class(mask, pred_mask)
    dice = dice_per_class(mask, pred_mask)
    print("Random Image Dice:", dice)
    print("Random Image IoU:", iou)

    visualize_predictions(img[0], mask, pred_mask[0], title="Random Image Prediction")

# Test the model
def test_model():
    test_loss, test_acc = model.evaluate(test_dataset)
    mean_ious, mean_dices = [], []
    per_class_iou = np.zeros(NUM_CLASSES)
    per_class_dice = np.zeros(NUM_CLASSES)
    sample_count = 0

    for img, mask in test_dataset:
        pred = model.predict(img)
        iou = iou_per_class(mask, pred)
        dice = dice_per_class(mask, pred)

        per_class_iou += np.array(iou)
        per_class_dice += np.array(dice)
        sample_count += 1

    per_class_iou /= sample_count
    per_class_dice /= sample_count

    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    for i in range(NUM_CLASSES):
        print(f"Class {i}: Dice = {per_class_dice[i]:.4f}, IoU = {per_class_iou[i]:.4f}")

    with open('test_results.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_acc:.4f}\n\n")
        for i in range(NUM_CLASSES):
            f.write(f"Class {i}: Dice = {per_class_dice[i]:.4f}, IoU = {per_class_iou[i]:.4f}\n")

    if np.mean(per_class_iou) > 0.7:
        print("✅ Model is performing well with high IoU.")
    elif np.mean(per_class_iou) > 0.5:
        print("⚠️ Model is decent but needs improvement.")
    else:
        print("❌ Model performance is low, consider tuning.")

    test_random_image()

# Recompile the model with Dice Loss
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

test_model()
