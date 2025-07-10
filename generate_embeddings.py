import numpy as np
from keras_facenet import FaceNet

# ========== CONFIG ==========
DATASET_PATH = "face_dataset.npz"
OUTPUT_PATH = "face_embeddings.npz"
# ============================

# ---------- Step 1: Load FaceNet ----------
print("📥 Loading FaceNet model...")
embedder = FaceNet()

# ---------- Step 2: Define get_embedding function ----------
def get_embedding(model, face):
    # Convert to float32
    face = face.astype('float32')
    # Standardize
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    # Expand dimensions to [1, 160, 160, 3]
    sample = np.expand_dims(face, axis=0)
    # Get embedding
    embedding = model.embeddings(sample)
    return embedding[0]

# ---------- Step 3: Load face dataset (.npz) ----------
print(f"📂 Loading dataset from {DATASET_PATH}...")
data = np.load(DATASET_PATH, allow_pickle=True)
trainX, trainy = data['trainX'], data['trainy']
testX, testy = data['testX'], data['testy']
print(f"✔️ Loaded {len(trainX)} train faces, {len(testX)} test faces.")

# ---------- Step 4: Generate embeddings ----------
print("🔁 Generating embeddings for training set...")
train_embeddings = np.array([get_embedding(embedder, face) for face in trainX])
print("🔁 Generating embeddings for test set...")
test_embeddings = np.array([get_embedding(embedder, face) for face in testX])

# ---------- Step 5: Save to .npz ----------
np.savez_compressed(
    OUTPUT_PATH,
    trainX=train_embeddings, trainy=trainy,
    testX=test_embeddings, testy=testy
)
print(f"✅ Embeddings saved to {OUTPUT_PATH}")
