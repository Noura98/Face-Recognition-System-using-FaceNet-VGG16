import streamlit as st
import numpy as np
from PIL import Image
import joblib
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import cv2


# ----- Load FaceNet model and classifier -----
@st.cache_resource
def load_facenet_model():
    return FaceNet().model


@st.cache_resource
def load_models():
    embedder = FaceNet()  # FaceNet embedding model
    classifier = joblib.load("knn_classifier.joblib")  # Your SVM model
    label_encoder = joblib.load("label_encoder.joblib")  # Your label encoder
    return embedder, classifier, label_encoder


embedder, model, label_encoder = load_models()

facenet_model = load_facenet_model()
detector = MTCNN()


# ----- Extract and align face -----
def extract_aligned_face(image_pil, required_size=(160, 160)):
    image = image_pil.convert('RGB')
    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None

    keypoints = results[0]['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # Compute angle between eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

    # Rotate image around the eye center
    aligned = image.rotate(-angle, center=eye_center)

    # Crop the face
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = aligned.crop((x1, y1, x2, y2)).resize(required_size)

    return np.asarray(face)


# ----- Streamlit UI -----
st.title("🔎 Face Recognition App")
st.write("Upload a photo to predict the person's identity using FaceNet + SVM Classifier")

uploaded_file = st.file_uploader("📁 Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Extract face
    face_array = extract_aligned_face(image)

    if face_array is None:
        st.error("❌ No face detected in the image.")
    else:
        # Normalize face
        face_array = face_array.astype('float32')
        mean, std = face_array.mean(), face_array.std()
        face_array = (face_array - mean) / std

        # Get FaceNet embedding
        embedding = facenet_model.predict(np.expand_dims(face_array, axis=0))[0]

        # Predict with classifier
        pred_encoded = model.predict([embedding])[0]
        pred_name = label_encoder.inverse_transform([pred_encoded])[0]
        pred = label_encoder.inverse_transform([pred_encoded])[0]
        proba = model.predict_proba([embedding])[0]
        confidence = np.max(proba) * 100

        st.success(f"✅ Predicted Identity: **{pred_name}**")
        st.info(f"🔒 Confidence: {confidence:.2f}%")
