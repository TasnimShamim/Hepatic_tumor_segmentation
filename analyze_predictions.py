import numpy as np
import matplotlib.pyplot as plt

# Optional: set to match your color scheme
colors = {
    0: (0, 0, 0),         # background - black
    1: (0, 255, 0),       # vessels - green
    2: (255, 0, 0),       # tumor - red
}

def colorize(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls, color in colors.items():
        color_mask[mask == cls] = color
    return color_mask

def visualize_batch(preds, truths, batch_size=5, start_idx=0):
    plt.figure(figsize=(12, 4 * batch_size))
    for i in range(batch_size):
        idx = start_idx + i
        pred = preds[idx]
        truth = truths[idx]

        plt.subplot(batch_size, 2, 2*i+1)
        plt.imshow(colorize(truth))
        plt.title(f"Ground Truth {idx}")
        plt.axis('off')

        plt.subplot(batch_size, 2, 2*i+2)
        plt.imshow(colorize(pred))
        plt.title(f"Prediction {idx}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def compute_metrics_per_class(preds, truths, num_classes=3):
    iou_per_class = np.zeros(num_classes)
    dice_per_class = np.zeros(num_classes)

    for cls in range(num_classes):
        TP = np.sum((preds == cls) & (truths == cls))
        FP = np.sum((preds == cls) & (truths != cls))
        FN = np.sum((preds != cls) & (truths == cls))

        iou = TP / (TP + FP + FN + 1e-6)
        dice = 2 * TP / (2 * TP + FP + FN + 1e-6)

        iou_per_class[cls] = iou
        dice_per_class[cls] = dice

    print("\n📈 Batch Evaluation Metrics:")
    for cls in range(num_classes):
        print(f"Class {cls}: IoU = {iou_per_class[cls]:.4f}, Dice = {dice_per_class[cls]:.4f}")

if __name__ == "__main__":
    preds = np.load("predicted_masks.npy")
    truths = np.load("ground_truth_masks.npy")

    visualize_batch(preds, truths, batch_size=5, start_idx=0)
    compute_metrics_per_class(preds, truths)
