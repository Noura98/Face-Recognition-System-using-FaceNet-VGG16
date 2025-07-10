import cv2
import os
import time
import shutil
from simple_image_download import simple_image_download



# Constants
DATASET_DIR = "dataset"
IMAGE_SIZE = (160, 160)
NUM_IMAGES = 40
DELAY_BETWEEN_CAPTURES = 0.7  # in seconds

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ---------- FUNCTION 1: Capture images using webcam ----------
def collect_from_webcam(person_name):
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print("\n📸 Starting webcam capture. Move your face slightly between shots.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, IMAGE_SIZE)
            count += 1
            file_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face)
            print(f"[{count}/{NUM_IMAGES}] Saved: {file_path}")
            time.sleep(DELAY_BETWEEN_CAPTURES)

        cv2.imshow("Webcam Face Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= NUM_IMAGES:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam capture finished.")


# ---------- FUNCTION 2: Download celebrity images ----------
def collect_from_internet(name):
    folder_name = name.replace(" ", "_").lower()
    person_dir = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(person_dir, exist_ok=True)

    print(f"\n🌐 Downloading images for: {name}")
    response = simple_image_download()
    response().download(name, NUM_IMAGES)

    # Path where images are downloaded
    raw_folder = os.path.join("simple_image_download", name)

    image_count = 0
    for file in os.listdir(raw_folder):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(raw_folder, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, IMAGE_SIZE)
            image_count += 1
            file_path = os.path.join(person_dir, f"{image_count}.jpg")
            cv2.imwrite(file_path, face)
            print(f"[{image_count}/{NUM_IMAGES}] Saved: {file_path}")
            break  # Only take the first face

        if image_count >= NUM_IMAGES:
            break

    shutil.rmtree("simple_image_download", ignore_errors=True)
    print("✅ Internet image collection finished.")



# ---------- MAIN MENU ----------
def main():
    print("\n💻 Face Dataset Builder")
    print("1. Collect face images from webcam")
    print("2. Collect face images of a famous person from the internet")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        name = input("Enter your name or person's name: ")
        collect_from_webcam(name.strip().replace(" ", "_").lower())

    elif choice == "2":
        name = input("Enter celebrity name to search and collect (e.g., 'Emma Watson'): ")
        collect_from_internet(name.strip())

    else:
        print("❌ Invalid choice. Please run again.")


if __name__ == "__main__":
    main()
