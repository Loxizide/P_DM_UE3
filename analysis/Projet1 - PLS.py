import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

# Charger les données
data_path = 'C:/Users/grego/Documents/Supagro - 2A/Data Manager/UE3/combined_data.csv'
df = pd.read_csv(data_path)

# Séparer les features (X) et la cible (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encodage des classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tester plusieurs nombres de composantes PLS
component_range = list(range(2, 30))
scores = []

for n in component_range:
    pls = PLSRegression(n_components=n)
    X_train_pls = pls.fit_transform(X_train_scaled, y_train)[0]
    clf = RandomForestClassifier(random_state=42)
    cv_score = cross_val_score(clf, X_train_pls, y_train, cv=5, scoring='accuracy').mean()
    scores.append(cv_score)

# Affichage de la courbe
plt.figure(figsize=(10, 6))
plt.plot(component_range, scores, marker='o')
plt.xlabel('Nombre de composantes PLS')
plt.ylabel('Accuracy moyenne (CV)')
plt.title('Choix du nombre optimal de composantes PLS')
plt.grid(True)
plt.tight_layout()
plt.show()

# Choix du meilleur
best_n = component_range[np.argmax(scores)]
print(f"\nNombre optimal de composantes PLS : {best_n}")

# Entraînement final avec ce nombre de composantes
pls_final = PLSRegression(n_components=best_n)
X_train_pls_final = pls_final.fit_transform(X_train_scaled, y_train)[0]
X_test_pls_final = pls_final.transform(X_test_scaled)

clf_final = RandomForestClassifier(random_state=42)
clf_final.fit(X_train_pls_final, y_train)
y_pred = clf_final.predict(X_test_pls_final)

# Résultats
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(f"Accuracy test : {accuracy_score(y_test, y_pred):.2f}")

# Matrice de confusion
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=label_encoder.classes_, cmap='Blues'
)
plt.title("Matrice de confusion - PLS + RandomForest")
plt.tight_layout()
plt.show()

# 🆕 VISUALISATION DES DEUX PREMIÈRES COMPOSANTES PLS
X_pls_2D = pls_final.transform(X_scaled := scaler.fit_transform(X))
y_full_encoded = label_encoder.transform(y)

plt.figure(figsize=(10, 7))
for class_index in np.unique(y_full_encoded):
    plt.scatter(
        X_pls_2D[y_full_encoded == class_index, 0],
        X_pls_2D[y_full_encoded == class_index, 1],
        label=label_encoder.inverse_transform([class_index])[0],
        alpha=0.7
    )

plt.xlabel("Composante PLS 1")
plt.ylabel("Composante PLS 2")
plt.title("Projection des données sur les 2 premières composantes PLS")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

