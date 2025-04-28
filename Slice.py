import os
import nibabel as nib
import numpy as np
import cv2

# ==== Configuration ====
IMG_SAVE_DIR = '2D_Sliced_Images/'
MASK_SAVE_DIR = '2D_Sliced_Masks/'
IMG_SIZE = (256, 256)  # Resize if needed
CLASS_LABELS_TO_KEEP = [1, 2]  # Keep slices that contain vessels or tumors

os.makedirs(IMG_SAVE_DIR, exist_ok=True)
os.makedirs(MASK_SAVE_DIR, exist_ok=True)

# ==== Helper Functions ====
def normalize_image(img):
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    return (img * 255).astype(np.uint8)

def slice_and_filter(image_path, mask_path, base_filename):
    image = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    assert image.shape == mask.shape, "Image and mask dimensions must match"

    saved_count = 0
    for i in range(image.shape[2]):
        img_slice = image[:, :, i]
        mask_slice = mask[:, :, i]

        # Resize slices
        img_resized = cv2.resize(img_slice, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_slice, IMG_SIZE, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # Check if the slice contains class 1 or 2
        if not any(val in np.unique(mask_resized) for val in CLASS_LABELS_TO_KEEP):
            continue  # Skip slice if no relevant labels

        # Prepare filenames
        img_filename = os.path.join(IMG_SAVE_DIR, f"{base_filename}_slice_{i}.png")
        mask_filename = os.path.join(MASK_SAVE_DIR, f"{base_filename}_slice_{i}.png")

        # Save normalized image
        cv2.imwrite(img_filename, normalize_image(img_resized))

        # Save visual-friendly mask (for checking)
        mask_vis = np.zeros_like(mask_resized, dtype=np.uint8)
        mask_vis[mask_resized == 1] = 127
        mask_vis[mask_resized == 2] = 255
        cv2.imwrite(mask_filename, mask_vis)

        # Optional: Save raw mask as .npy for training
        # np.save(mask_filename.replace('.png', '.npy'), mask_resized)

        saved_count += 1

    return saved_count

# ==== Main Function ====
def process_all_volumes(image_dir, mask_dir):
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith('.nii.gz') and not f.startswith("._")
    ])

    total_saved = 0
    for filename in image_files:
        base = filename.replace('_0000.nii.gz', '').replace('.nii.gz', '')
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, base + '.nii.gz')  # Assumes masks share base name

        print(f"Processing {base}...")
        saved = slice_and_filter(image_path, mask_path, base)
        print(f" → Saved {saved} valid slices.")
        total_saved += saved

    print(f"\n✅ Total valid slices saved: {total_saved}")

# ==== Run ====
if __name__ == "__main__":
    IMAGE_DIR = "Task08_HepaticVessel/imagesTr"
    MASK_DIR = "Task08_HepaticVessel/labelsTr"
    process_all_volumes(IMAGE_DIR, MASK_DIR)
    print("All images and masks have been processed and saved.")
