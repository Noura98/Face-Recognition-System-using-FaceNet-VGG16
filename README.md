# Face-Recognition-System-using-FaceNet-SVM
This project is a real-time face recognition system built with FaceNet for face embedding and Support Vector Machine (SVM) for classification. It includes a Streamlit app for easy testing and interaction.
------------------------

🚀 How It Works
Face Detection
Detect faces from raw images using OpenCV or MTCNN.
Face Embedding
Convert faces to 128-D feature vectors using FaceNet (via keras-facenet).
----------------------------
📦 Requirements

streamlit
opencv-python
keras-facenet
numpy
scikit-learn
joblib
Pillow
---------------------
📸 Dataset Preparation
Choose 1 to 5 people

Collect 40 images per person:

30 for training → dataset/train/PersonName/

10 for testing → dataset/test/PersonName/

Face Extraction: Use HAAR cascade or MTCNN

Train SVM with the FaceNet embeddings

---------------------

