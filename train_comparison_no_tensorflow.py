"""
IDS Cloud - Comparaison ML (Random Forest) vs DL (MLP / Reseau de neurones)
Dataset : CSE-CIC-IDS2018 (genere sur infrastructure AWS / Amazon Cloud)
Version SANS TensorFlow -> fonctionne avec juste scikit-learn
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

np.random.seed(42)

# ---------------------------------------------------------
# 1. Chargement des donnees
#    Recherche automatique du fichier CSV dans le dossier du
#    script et tous ses sous-dossiers (peu importe ou tu l'as mis)
# ---------------------------------------------------------
CSV_NAME = "EDOS-CSE-CIC-IDS2018-dataset.csv"
BASE_DIR = Path(__file__).resolve().parent

print(f"Recherche de '{CSV_NAME}' dans : {BASE_DIR}  (et ses sous-dossiers)...")
matches = list(BASE_DIR.rglob(CSV_NAME))

if not matches:
    print(f"\n!!! ERREUR : impossible de trouver '{CSV_NAME}'.")
    print(f"Verifie que tu as bien extrait le zip et place le fichier .csv")
    print(f"quelque part dans ce dossier : {BASE_DIR}\n")
    print("Voici ce qui se trouve actuellement dans ce dossier :")
    for item in sorted(BASE_DIR.rglob("*")):
        print("  -", item.relative_to(BASE_DIR))
    raise SystemExit(1)

csv_path = matches[0]
print(f"Dataset trouve : {csv_path}\n")

print("Chargement du dataset CSE-CIC-IDS2018 (cloud / AWS)...")
df = pd.read_csv(csv_path)
print("Taille totale :", df.shape)
print(df["Attack"].value_counts())

# ---------------------------------------------------------
# 2. Pretraitement
# ---------------------------------------------------------
X = df.drop(columns=["Attack", "label"])
y = df["label"]  # 0 = Benign, 1 = Attaque

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")

results = {}
predictions = {}

# ---------------------------------------------------------
# 3. MODELE 1 : Random Forest (Machine Learning)
# ---------------------------------------------------------
print("\n=== Entrainement Random Forest (ML) ===")
rf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
t0 = time.time()
rf.fit(X_train_scaled, y_train)
rf_train_time = time.time() - t0

y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

results["Random Forest (ML)"] = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Precision": precision_score(y_test, y_pred_rf),
    "Recall": recall_score(y_test, y_pred_rf),
    "F1-score": f1_score(y_test, y_pred_rf),
    "AUC-ROC": roc_auc_score(y_test, y_proba_rf),
    "Temps entrainement (s)": rf_train_time,
}
cm_rf = confusion_matrix(y_test, y_pred_rf)
tn, fp, fn, tp = cm_rf.ravel()
results["Random Forest (ML)"]["FPR"] = fp / (fp + tn)
predictions["Random Forest (ML)"] = (y_pred_rf, y_proba_rf, cm_rf)

print(f"Termine en {rf_train_time:.1f}s")
for k, v in results["Random Forest (ML)"].items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# ---------------------------------------------------------
# 4. MODELE 2 : MLP - Multi-Layer Perceptron (Deep Learning)
#    Reseau de neurones a 3 couches cachees (128, 64, 32 neurones)
#    C'est un vrai reseau de neurones profond (Deep Neural Network),
#    implemente directement avec scikit-learn (pas besoin de TensorFlow)
# ---------------------------------------------------------
print("\n=== Entrainement MLP / Reseau de neurones (DL) ===")

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=512,
    learning_rate_init=0.001,
    max_iter=60,
    early_stopping=True,
    n_iter_no_change=5,
    random_state=42,
    verbose=False,
)

t0 = time.time()
mlp.fit(X_train_scaled, y_train)
dnn_train_time = time.time() - t0

y_pred_dnn = mlp.predict(X_test_scaled)
y_proba_dnn = mlp.predict_proba(X_test_scaled)[:, 1]

results["MLP / Reseau de neurones (DL)"] = {
    "Accuracy": accuracy_score(y_test, y_pred_dnn),
    "Precision": precision_score(y_test, y_pred_dnn),
    "Recall": recall_score(y_test, y_pred_dnn),
    "F1-score": f1_score(y_test, y_pred_dnn),
    "AUC-ROC": roc_auc_score(y_test, y_proba_dnn),
    "Temps entrainement (s)": dnn_train_time,
}
cm_dnn = confusion_matrix(y_test, y_pred_dnn)
tn, fp, fn, tp = cm_dnn.ravel()
results["MLP / Reseau de neurones (DL)"]["FPR"] = fp / (fp + tn)
predictions["MLP / Reseau de neurones (DL)"] = (y_pred_dnn, y_proba_dnn, cm_dnn)

print(f"Termine en {dnn_train_time:.1f}s (nombre d'iterations: {mlp.n_iter_})")
for k, v in results["MLP / Reseau de neurones (DL)"].items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# ---------------------------------------------------------
# 5. Sauvegarde des resultats
# ---------------------------------------------------------
with open("results_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

rows = []
for model_name, metrics in results.items():
    row = {"Modele": model_name}
    row.update(metrics)
    rows.append(row)
comp_df = pd.DataFrame(rows)
comp_df.to_csv("results_comparison.csv", index=False)
print("\n", comp_df.round(4).to_string(index=False))

# ---------------------------------------------------------
# 6. Graphiques comparatifs
# ---------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
colors = {"Random Forest (ML)": "#2563eb", "MLP / Reseau de neurones (DL)": "#dc2626"}

# --- Bar chart comparatif des metriques ---
metric_names = ["Accuracy", "Precision", "Recall", "F1-score", "FPR", "AUC-ROC"]
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(metric_names))
width = 0.35
for i, (model_name, metrics) in enumerate(results.items()):
    values = [metrics[m] for m in metric_names]
    bars = ax.bar(x + i*width - width/2, values, width, label=model_name, color=colors[model_name])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("Comparaison des metriques : Random Forest (ML) vs MLP (DL)\nDataset CSE-CIC-IDS2018 (cloud AWS)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("comparison_metrics_barchart.png", dpi=150)
plt.close()

# --- Courbes ROC comparees ---
fig, ax = plt.subplots(figsize=(6, 5))
for model_name, (y_pred, y_proba, cm) in predictions.items():
    fpr_c, tpr_c, _ = roc_curve(y_test, y_proba)
    auc = results[model_name]["AUC-ROC"]
    ax.plot(fpr_c, tpr_c, label=f"{model_name} (AUC={auc:.3f})", color=colors[model_name], linewidth=2)
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Aleatoire")
ax.set_xlabel("Taux de faux positifs (FPR)")
ax.set_ylabel("Taux de vrais positifs (TPR)")
ax.set_title("Courbes ROC comparees - CSE-CIC-IDS2018")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("comparison_roc_curves.png", dpi=150)
plt.close()

# --- Matrices de confusion cote a cote ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (model_name, (y_pred, y_proba, cm)) in zip(axes, predictions.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Attaque"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(model_name)
plt.tight_layout()
plt.savefig("comparison_confusion_matrices.png", dpi=150)
plt.close()

# --- Courbe d'apprentissage du MLP (perte par iteration) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(mlp.loss_curve_, color="#dc2626", linewidth=2)
ax.set_title("Courbe d'apprentissage - MLP (perte par iteration)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
plt.tight_layout()
plt.savefig("dnn_learning_curves.png", dpi=150)
plt.close()

print("\nFichiers generes : results_comparison.csv/json, comparison_metrics_barchart.png, "
      "comparison_roc_curves.png, comparison_confusion_matrices.png, dnn_learning_curves.png")
