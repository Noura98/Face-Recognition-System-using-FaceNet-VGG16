import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from collections import Counter
from scipy.spatial.distance import cosine

# 1. Load embeddings and labels
data = np.load('faces-embeddings.npz')
trainX, trainy = data['trainX'], data['trainy']
testX, testy = data['testX'], data['testy']

# 2. Encode labels
label_encoder = LabelEncoder()
trainy_enc = label_encoder.fit_transform(trainy)
testy_enc = label_encoder.transform(testy)

print("Train labels (raw):", Counter(trainy))
print("Train labels (encoded):", Counter(trainy_enc))
print("Test labels (encoded):", Counter(testy_enc))

# 3. Normalize embeddings
normalizer = Normalizer(norm='l2')
trainX_norm = normalizer.fit_transform(trainX)
testX_norm = normalizer.transform(testX)

# 4. Distance examples
dist_same = cosine(trainX_norm[0], trainX_norm[1])
dist_diff = cosine(trainX_norm[0], trainX_norm[30])

print("✅ Distance between 2 images of the same person:", dist_same)
print("❌ Distance between 2 different people:", dist_diff)

# 5. Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')  
knn.fit(trainX_norm, trainy_enc)

# 6. Predict
y_pred_knn = knn.predict(testX_norm)

# 7. Evaluation
print(classification_report(testy_enc, y_pred_knn, target_names=label_encoder.classes_))

# 8. Save models
joblib.dump(knn, 'knn_classifier.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')  # You can reuse the same
joblib.dump(normalizer, 'normalizer.joblib')

print("KNN model, label encoder, and normalizer saved!")
