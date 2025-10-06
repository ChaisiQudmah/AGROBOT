import os
import random
import shutil


RAW_DATASET_DIR = "dataset"
PROCESSED_DATASET_DIR = "processed_dataset"
TRAIN_SPLIT = 0.8
CLASSES_TO_SELECT = 8
IMAGES_PER_CLASS = 1000

def main():
    
    if os.path.exists(PROCESSED_DATASET_DIR):
        shutil.rmtree(PROCESSED_DATASET_DIR)
    os.makedirs(PROCESSED_DATASET_DIR)

  
    all_classes = [d for d in os.listdir(RAW_DATASET_DIR) if os.path.isdir(os.path.join(RAW_DATASET_DIR, d))]
    print(f"Total classes found: {len(all_classes)}")

    
    valid_classes = []
    for cls in all_classes:
        cls_path = os.path.join(RAW_DATASET_DIR, cls)
        if len(os.listdir(cls_path)) >= IMAGES_PER_CLASS:
            valid_classes.append(cls)

    print(f"Classes with at least {IMAGES_PER_CLASS} images: {len(valid_classes)}")

    
    if len(valid_classes) < CLASSES_TO_SELECT:
        print(" Not enough classes with 1000+ images. Please reduce CLASSES_TO_SELECT.")
        return

    selected_classes = random.sample(valid_classes, CLASSES_TO_SELECT)
    print(f"Selected classes: {selected_classes}")

    for cls in selected_classes:
        cls_path = os.path.join(RAW_DATASET_DIR, cls)
        images = os.listdir(cls_path)

        
        random.shuffle(images)
        selected_images = images[:IMAGES_PER_CLASS]

        
        split_idx = int(IMAGES_PER_CLASS * TRAIN_SPLIT)
        train_imgs = selected_images[:split_idx]
        test_imgs = selected_images[split_idx:]

        
        train_dir = os.path.join(PROCESSED_DATASET_DIR, "train", cls)
        test_dir = os.path.join(PROCESSED_DATASET_DIR, "test", cls)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        
        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, img))
        for img in test_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(test_dir, img))

        print(f"{cls}: {len(train_imgs)} train, {len(test_imgs)} test")

    print("\n Processing complete! Use processed_dataset/ for training.")

if __name__ == "__main__":
    main()
