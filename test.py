import os
import cv2
import shutil
from icrawler.builtin import BingImageCrawler
import random
from PIL import Image
import imagehash
import numpy as np

# ====== CONFIG ======
IMAGE_SIZE = (160, 160)
NUM_IMAGES = 40
TRAIN_SPLIT = 30
CASCADE_PATH = "haarcascade_frontalface_default.xml"
DATASET_DIR = "dataset"
# ====================

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def extract_face_from_saved(img, size=IMAGE_SIZE):
    # These are already cropped faces; just ensure they are resized properly
    return cv2.resize(img, size)

def capture_faces_from_camera(person_name):
    print(f"\n📸 Capturing images for: {person_name}")
    cap = cv2.VideoCapture(0)
    saved = 0
    images = []

    while saved < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            break
        face = extract_face_from_saved(frame)
        if face is not None:
            images.append(face)
            saved += 1
            print(f"Captured [{saved}/{NUM_IMAGES}]")
            cv2.imshow('Captured Face', face)
        else:
            cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return images

def download_faces_from_internet(person_name):
    print(f"\n🌐 Downloading images for: {person_name}")
    base_dir = f"temp_images/{person_name.replace(' ', '_')}"
    os.makedirs(base_dir, exist_ok=True)

    collected_faces = []
    collected_hashes = set()  # Store perceptual hashes
    attempt = 0
    max_attempts = 5  # Try crawling multiple times
    images_to_fetch_each_time = 60

    while len(collected_faces) < NUM_IMAGES and attempt < max_attempts:
        print(f"🔁 Attempt {attempt+1} to collect faces...")

        crawler = BingImageCrawler(storage={"root_dir": base_dir})
        crawler.crawl(keyword=person_name, max_num=images_to_fetch_each_time)

        for file in os.listdir(base_dir):
            path = os.path.join(base_dir, file)
            img = cv2.imread(path)
            if img is None:
                continue

            face = extract_face_from_saved(img)
            if face is None:
                continue

            # Convert face to PIL image for hashing
            pil_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_hash = imagehash.phash(pil_image)

            # Check if similar hash already exists
            is_duplicate = any(abs(face_hash - existing_hash) <= 5 for existing_hash in collected_hashes)
            if is_duplicate:
                continue

            # Add face and its hash
            collected_faces.append(face)
            collected_hashes.add(face_hash)
            print(f"✅ Total unique faces: {len(collected_faces)}/{NUM_IMAGES}")

            if len(collected_faces) >= NUM_IMAGES:
                break

        # Clean up before next attempt
        shutil.rmtree(base_dir, ignore_errors=True)
        os.makedirs(base_dir, exist_ok=True)
        attempt += 1

    shutil.rmtree("temp_images", ignore_errors=True)

    if len(collected_faces) >= NUM_IMAGES:
        print(f"✅ Successfully collected {NUM_IMAGES} unique face images for {person_name}")
        return collected_faces[:NUM_IMAGES]
    else:
        print(f"❌ Could not collect enough unique faces for {person_name}. Only {len(collected_faces)} collected.")
        return []




def save_faces(images, person_name):
    train_dir = os.path.join(DATASET_DIR, "train", person_name)
    test_dir = os.path.join(DATASET_DIR, "test", person_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for i in range(NUM_IMAGES):  # Loop through all 40 images
        face = images[i]
        if i < TRAIN_SPLIT:  # First 30 images → train
            path = os.path.join(train_dir, f"{i+1}.jpg")
        else:  # Remaining 10 images → test
            path = os.path.join(test_dir, f"{i+1-TRAIN_SPLIT}.jpg")
        cv2.imwrite(path, face)

def load_faces(directory):
    faces = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        face = extract_face_from_saved(img)  # ⬅️ NOTE ici !
        if face is not None:
            faces.append(face)
    return faces


def load_dataset(parent_dir):
    X, y = [], []
    for label in os.listdir(parent_dir):
        path = os.path.join(parent_dir, label)
        if not os.path.isdir(path):
            continue
        faces = load_faces(path)
        X.extend(faces)
        y.extend([label] * len(faces))
    return np.array(X), np.array(y)

def save_dataset_to_npz(train_dir, test_dir, output_file="face_dataset.npz"):
    print("\n📦 Loading and saving dataset...")

    trainX, trainy = load_dataset(train_dir)
    testX, testy = load_dataset(test_dir)

    print(f"✔️ Train set: {len(trainX)} faces")
    print(f"✔️ Test set: {len(testX)} faces")

    np.savez_compressed(output_file, trainX=trainX, trainy=trainy, testX=testX, testy=testy)
    print(f"💾 Dataset saved to {output_file}")

def main():
    print("Face Dataset Builder")
    print("---------------------")
    num_people = int(input("How many people do you want to add? (1–5): "))
    for i in range(num_people):
        print(f"\nPerson {i+1}:")
        person_name = input("Enter name (no spaces): ").strip()
        choice = input("Choose data source: [1] Camera, [2] Internet → ")
        if choice == "1":
            faces = capture_faces_from_camera(person_name)
        elif choice == "2":
            faces = download_faces_from_internet(person_name)
        else:
            print("Invalid choice. Skipping...")
            continue

        if len(faces) < NUM_IMAGES:
            print(f"⚠️ Only collected {len(faces)} faces. Skipping saving.")
            continue

        random.shuffle(faces)
        save_faces(faces, person_name)
        print(f"✅ Saved {NUM_IMAGES} faces for {person_name}")

    # save all into a compressed .npz
    save_dataset_to_npz(
        os.path.join(DATASET_DIR, "train"),
        os.path.join(DATASET_DIR, "test")
    )

if __name__ == "__main__":
    main()
