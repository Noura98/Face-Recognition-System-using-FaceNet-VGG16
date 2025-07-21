import numpy as np
from keras_facenet import FaceNet
from collections import Counter

# ---------- Step 1: Load dataset ----------
data = np.load('faces-dataset.npz', allow_pickle=True)
trainX, trainy = data['trainX'], data['trainy']
testX, testy = data['testX'], data['testy']

print("\n✅ Face count per class (train):", Counter(trainy))
print("✅ Face count per class (test):", Counter(testy))

# ---------- Step 2: Load FaceNet model ----------
embedder = FaceNet()

# ---------- Step 3: Embedding functions ----------
def get_embedding(model, face_pixels):
    return model.embeddings([face_pixels])[0]

def create_embeddings(model, faces_array):
    return np.asarray([get_embedding(model, face) for face in faces_array])

# ---------- Step 4: Create and Save Embeddings ----------
train_embeddings = create_embeddings(embedder, trainX)
test_embeddings = create_embeddings(embedder, testX)

np.savez_compressed('faces-embeddings.npz',
                    trainX=train_embeddings, trainy=trainy,
                    testX=test_embeddings, testy=testy)

print("✅ Embeddings saved in 'faces-embeddings.npz'")