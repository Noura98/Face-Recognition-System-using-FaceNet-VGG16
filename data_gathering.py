import os
import cv2
import urllib.request
from numpy import asarray
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt

detector = MTCNN()

# Step 0: Download HAAR cascade file if not present
CASCADE_PATH = "haarcascade_frontalface_default.xml"
haar_cascade_filename = "haarcascade_frontalface_default.xml"

if not os.path.exists(haar_cascade_filename):
    print("Downloading HAAR cascade model...")
    urllib.request.urlretrieve(CASCADE_PATH, haar_cascade_filename)
    print("Download completed.")

# Load HAAR cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_cascade_filename)

# Ensure debug_faces folder exists
os.makedirs("debug_faces", exist_ok=True)


def extract_aligned_face(filename, required_size=(160,160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None

    # Get landmarks
    keypoints = results[0]['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # Compute angle for rotation based on eye coordinates
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate around center between eyes to align
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)
    pil_image = Image.fromarray(pixels)
    pil_image = pil_image.rotate(-angle, center=eye_center)

    # Crop face box from rotated image
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    face = pil_image.crop((x1, y1, x2, y2))
    face = face.resize(required_size)

    return asarray(face)


def load_faces(directory, label_name, show_limit=3):
    faces = []
    shown = 0  # Counter for how many face images shown
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        face = extract_aligned_face(path)
        if face is not None:
            # Check shape
            if face.shape != (160, 160, 3):
                print(f"Invalid shape {face.shape} in file: {filename}, skipping.")
                continue

            # Optional: display first few faces per class
            if shown < show_limit:
                plt.imshow(face)
                plt.title(f"{label_name} - {filename}")
                plt.axis('off')
                plt.show()
                shown += 1

            # Optional: save extracted face
            save_path = os.path.join("debug_faces", f"{label_name}_{filename}")
            Image.fromarray(face).save(save_path)

            faces.append(face)
        else:
            print(f"Warning: No face found in {path}")
    return faces

def load_dataset(parent_directory):
    X, y = [], []
    for subdir in os.listdir(parent_directory):
        path = os.path.join(parent_directory, subdir)
        if not os.path.isdir(path):
            continue

        print(f"\n🔍 Loading faces for: {subdir}")
        faces = load_faces(path, subdir)
        labels = [subdir for _ in range(len(faces))]

        X.extend(faces)
        y.extend(labels)

    return asarray(X), y

if __name__ == "__main__":
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'

    print("\n📦 Loading training dataset...")
    trainX, trainy = load_dataset(train_dir)
    print(f"✅ Loaded {trainX.shape[0]} faces for training.")

    print("\n📦 Loading testing dataset...")
    testX, testy = load_dataset(test_dir)
    print(f"✅ Loaded {testX.shape[0]} faces for testing.")

    # Save datasets
    np.savez_compressed('faces-dataset.npz', trainX=trainX, trainy=trainy, testX=testX, testy=testy)
    print("\n💾 Saved datasets to faces-dataset.npz")
