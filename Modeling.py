import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
# 1. Load embeddings and labels
data = np.load('faces-embeddings.npz')
trainX, trainy = data['trainX'], data['trainy']
testX, testy = data['testX'], data['testy']

# 2. Encode labels
label_encoder = LabelEncoder()
trainy_enc = label_encoder.fit_transform(trainy)
testy_enc = label_encoder.transform(testy)
from collections import Counter
print("Train labels (raw):", Counter(trainy))
print("Train labels (encoded):", Counter(trainy_enc))
print("Test labels (encoded):", Counter(testy_enc))
# 3. Normalize embeddings (L2 norm)
normalizer = Normalizer(norm='l2')
trainX_norm = normalizer.fit_transform(trainX)
testX_norm = normalizer.transform(testX)

from scipy.spatial.distance import cosine
import numpy as np

# Fonction de comparaison
def compare_distance(sampleA, sampleB):
    return cosine(sampleA, sampleB)

# Afficher les distances entre deux embeddings d'une même personne (Angelina Jolie)
dist1 = compare_distance(trainX_norm[0], trainX_norm[1])

# Afficher la distance entre une image d'Angelina Jolie et une image d'une autre personne (ex. Brad Pitt)
dist2 = compare_distance(trainX_norm[0], trainX_norm[30])  # 30 → début de Brad Pitt si l’ordre est respecté

print("✅ Distance entre 2 images d'une même personne :", dist1)
print("❌ Distance entre 2 personnes différentes :", dist2)

# 4. Train SVM with linear kernel and probability estimates enabled
svm = SVC(kernel='rbf', probability=True, C=10)
svm.fit(trainX_norm, trainy_enc)

# 5. Predict on test set
y_pred_svm = svm.predict(testX_norm)

# 6. Classification report
print(classification_report(testy_enc, y_pred_svm, target_names=label_encoder.classes_))

# 7. Save models for later use
joblib.dump(svm, 'svm_classifier.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
joblib.dump(normalizer, 'normalizer.joblib')

print("SVM model, label encoder, and normalizer saved!")