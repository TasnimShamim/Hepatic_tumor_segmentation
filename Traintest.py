import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# Configurations
IMAGE_PATH = '2D_Sliced_Images/'
MASK_PATH = '2D_Sliced_Masks/'
IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 8
EPOCHS = 50
NUM_CLASSES = 3

# Load file paths
image_files = sorted([os.path.join(IMAGE_PATH, f) for f in os.listdir(IMAGE_PATH) if f.endswith('.png')])
mask_files = sorted([os.path.join(MASK_PATH, f) for f in os.listdir(MASK_PATH) if f.endswith('.png')])

# Split dataset
train_images, test_images, train_masks, test_masks = train_test_split(image_files, mask_files, test_size=0.2, random_state=42)
train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, test_size=0.25, random_state=42)

# ✅ Preview one random CT scan and ground truth mask
def preview_loaded_image_and_mask(image_path, mask_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Convert to class labels
    mask_class = np.zeros_like(mask)
    mask_class[mask == 0] = 0
    mask_class[mask == 128] = 1
    mask_class[mask == 255] = 2

    # Color map: 0 = black, 1 = cyan, 2 = red
    color_mask = np.zeros((*mask_class.shape, 3), dtype=np.uint8)
    color_mask[mask_class == 1] = [0, 255, 255]  # Cyan
    color_mask[mask_class == 2] = [255, 0, 0]    # Red

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("CT Scan")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(color_mask)
    plt.title("Ground Truth Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 🔍 Preview one random pair
random_idx = random.randint(0, len(train_images) - 1)
preview_loaded_image_and_mask(train_images[random_idx], train_masks[random_idx])


# Data Augmentation
def augment_image_and_mask(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    if tf.random.uniform(()) > 0.5:
        k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        mask = tf.image.rot90(mask, k)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.1)
    return image, mask

# Data Generator
def data_generator(image_list, mask_list, augment=False):
    for img_path, mask_path in zip(image_list, mask_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img)
        mask = np.array(mask)

        mask_class = np.zeros_like(mask)
        mask_class[mask == 0] = 0
        mask_class[mask == 128] = 1
        mask_class[mask == 255] = 2

        img = np.expand_dims(img, axis=-1)
        mask_onehot = tf.keras.utils.to_categorical(mask_class, NUM_CLASSES)

        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        mask_tensor = tf.convert_to_tensor(mask_onehot, dtype=tf.float32)

        if augment:
            img_tensor, mask_tensor = augment_image_and_mask(img_tensor, mask_tensor)

        yield img_tensor, mask_tensor

def create_dataset(image_list, mask_list, augment=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(image_list, mask_list, augment),
        output_signature=(
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES), dtype=tf.float32)
        )
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Datasets
train_dataset = create_dataset(train_images, train_masks, augment=True)
val_dataset = create_dataset(val_images, val_masks, augment=False)
test_dataset = create_dataset(test_images, test_masks, augment=False)

# U-Net Model
def build_unet_model(input_shape, num_classes):
    inputs = layers.Input(input_shape)

    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)

    u1 = layers.UpSampling2D((2, 2))(b)
    u1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    u2 = layers.UpSampling2D((2, 2))(u1)
    u2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    u3 = layers.UpSampling2D((2, 2))(u2)
    u3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u3)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(u3)

    return keras.Model(inputs, outputs) 

# Dice Loss
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / (denominator + 1e-6)

# Evaluation Metrics
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

# ✅ Visualization with class color legend
def visualize_predictions(img, mask, pred_mask, title="Prediction Visualization"):
    img = np.squeeze(img, axis=-1)
    true_mask = np.argmax(mask, axis=-1)
    pred_mask = np.argmax(pred_mask, axis=-1)

    custom_cmap = ListedColormap(["black", "cyan", "red"])
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='Background (0)'),
        Patch(facecolor='cyan', edgecolor='cyan', label='Vessels (1)'),
        Patch(facecolor='red', edgecolor='red', label='Tumor (2)')
    ]

    plt.figure(figsize=(12, 4))
    plt.suptitle(title, fontsize=14)

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("CT Scan")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, cmap=custom_cmap, vmin=0, vmax=2)
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap=custom_cmap, vmin=0, vmax=2)
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(-0.1, -0.35), ncol=3)
    plt.tight_layout(rect=[0, 0.1, 1, 0.92])
    plt.show()

# Plot training curves
def plot_training_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.show()

# Test on one random image
def test_random_image(model):
    idx = random.randint(0, len(test_images) - 1)
    img = cv2.imread(test_images[idx], cv2.IMREAD_GRAYSCALE) / 255.0
    mask = cv2.imread(test_masks[idx], cv2.IMREAD_GRAYSCALE)

    mask_class = np.zeros_like(mask)
    mask_class[mask == 0] = 0
    mask_class[mask == 128] = 1
    mask_class[mask == 255] = 2

    img = np.expand_dims(img, axis=(0, -1))
    mask = tf.keras.utils.to_categorical(mask_class, NUM_CLASSES)
    pred_mask = model.predict(img)

    iou = iou_per_class(mask, pred_mask)
    dice = dice_per_class(mask, pred_mask)
    print("Random Image Dice:", dice)
    print("Random Image IoU:", iou)

    visualize_predictions(img[0], mask, pred_mask[0], title="Random Image Prediction")

# Evaluate on full test set with classwise results
def test_model(model):
    test_loss, test_acc = model.evaluate(test_dataset)
    per_class_iou = np.zeros(NUM_CLASSES)
    per_class_dice = np.zeros(NUM_CLASSES)
    per_class_accuracy = np.zeros(NUM_CLASSES)
    total_class_pixels = np.zeros(NUM_CLASSES)
    sample_count = 0

    all_preds = []
    all_truths = []

    for img_batch, mask_batch in test_dataset:
        pred_batch = model.predict(img_batch)
        pred_labels = tf.argmax(pred_batch, axis=-1).numpy()
        true_labels = tf.argmax(mask_batch, axis=-1).numpy()

        all_preds.append(pred_labels)
        all_truths.append(true_labels)

        for i in range(NUM_CLASSES):
            class_mask = (true_labels == i)
            correct_predictions = np.sum((pred_labels == i) & class_mask)
            total_pixels = np.sum(class_mask)

            per_class_accuracy[i] += correct_predictions
            total_class_pixels[i] += total_pixels

        iou = iou_per_class(mask_batch, pred_batch)
        dice = dice_per_class(mask_batch, pred_batch)

        per_class_iou += np.array(iou)
        per_class_dice += np.array(dice)
        sample_count += 1

    per_class_iou /= sample_count
    per_class_dice /= sample_count
    per_class_accuracy = per_class_accuracy / (total_class_pixels + 1e-6)

    # Save all predicted and ground truth masks
    all_preds = np.concatenate(all_preds, axis=0)
    all_truths = np.concatenate(all_truths, axis=0)
    np.save("predicted_masks.npy", all_preds)
    np.save("ground_truth_masks.npy", all_truths)

    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    for i in range(NUM_CLASSES):
        print(f"Class {i}: Dice = {per_class_dice[i]:.4f}, IoU = {per_class_iou[i]:.4f}, Accuracy = {per_class_accuracy[i]:.4f}")

    with open('test_results.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_acc:.4f}\n\n")
        for i in range(NUM_CLASSES):
            f.write(f"Class {i}: Dice = {per_class_dice[i]:.4f}, IoU = {per_class_iou[i]:.4f}, Accuracy = {per_class_accuracy[i]:.4f}\n")

    mean_iou = np.mean(per_class_iou)
    if mean_iou > 0.7:
        print("✅ Model is performing well with high IoU.")
    elif mean_iou > 0.5:
        print("⚠️ Model is decent but needs improvement.")
    else:
        print("❌ Model performance is low, consider tuning.")

    test_random_image(model)



