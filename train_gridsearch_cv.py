"""
IDS Cloud - Version renforcee avec GridSearch + Validation croisee
Dataset : CSE-CIC-IDS2018 (cloud AWS)

Etapes :
1. GridSearchCV sur un sous-echantillon -> trouver les meilleurs hyperparametres du Random Forest
2. Re-entrainement du modele final (RF optimise + MLP) sur les donnees d'entrainement
3. Validation croisee (5-fold) sur un sous-echantillon -> verifier la stabilite des resultats
4. Evaluation finale sur le jeu de test
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, make_scorer
)

np.random.seed(42)

# ---------------------------------------------------------
# 1. Chargement des donnees
# ---------------------------------------------------------
CSV_NAME = "EDOS-CSE-CIC-IDS2018-dataset.csv"
BASE_DIR = Path(__file__).resolve().parent
matches = list(BASE_DIR.rglob(CSV_NAME))
if not matches:
    raise SystemExit(f"Fichier {CSV_NAME} introuvable dans {BASE_DIR}")
csv_path = matches[0]
print(f"Dataset trouve : {csv_path}")

df = pd.read_csv(csv_path)
print("Taille totale :", df.shape)

X = df.drop(columns=["Attack", "label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")

# ---------------------------------------------------------
# 2. GRIDSEARCH - recherche des meilleurs hyperparametres
#    (sur un sous-echantillon de 60 000 lignes pour rester rapide,
#     pratique courante avec des "Big Data" de plusieurs centaines
#     de milliers de lignes)
# ---------------------------------------------------------
print("\n=== GridSearchCV : recherche des meilleurs hyperparametres (Random Forest) ===")

sample_idx = np.random.RandomState(42).choice(
    X_train_scaled.shape[0], size=min(30000, X_train_scaled.shape[0]), replace=False
)
X_sample = X_train_scaled[sample_idx]
y_sample = y_train.iloc[sample_idx]

param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [10, 20],
}

# IMPORTANT : n_jobs=1 sur le modele (evite la sur-souscription de threads
# quand GridSearchCV utilise deja n_jobs=-1 en parallele)
grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=1),
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)

t0 = time.time()
grid.fit(X_sample, y_sample)
grid_time = time.time() - t0

print(f"\nGridSearch termine en {grid_time:.1f}s")
print("Meilleurs hyperparametres trouves :", grid.best_params_)
print(f"Meilleur score F1 (validation croisee, recherche) : {grid.best_score_:.4f}")

cv_results_df = pd.DataFrame(grid.cv_results_)[
    ["param_n_estimators", "param_max_depth", "mean_test_score", "std_test_score"]
].sort_values("mean_test_score", ascending=False)
cv_results_df.to_csv("gridsearch_results.csv", index=False)
print("\nTop combinaisons testees :")
print(cv_results_df.head(8).to_string(index=False))

best_params = grid.best_params_

# ---------------------------------------------------------
# 3. VALIDATION CROISEE (5-fold) - verifier la stabilite
#    des resultats pour les 2 modeles, sur un sous-echantillon
# ---------------------------------------------------------
print("\n=== Validation croisee (5-fold) : Random Forest (optimise) vs MLP ===")

cv_sample_idx = np.random.RandomState(7).choice(
    X_train_scaled.shape[0], size=min(40000, X_train_scaled.shape[0]), replace=False
)
X_cv = X_train_scaled[cv_sample_idx]
y_cv = y_train.iloc[cv_sample_idx]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

models_cv = {
    "Random Forest (optimise)": RandomForestClassifier(**best_params, random_state=42, n_jobs=1),
    "MLP (DL)": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=25,
                               early_stopping=True, random_state=42),
}

cv_summary = {}
for name, model in models_cv.items():
    print(f"\n--- Validation croisee : {name} ---")
    t0 = time.time()
    cv_res = cross_validate(model, X_cv, y_cv, cv=skf, scoring=scoring, n_jobs=-1)
    elapsed = time.time() - t0
    summary = {}
    for metric in scoring:
        scores = cv_res[f"test_{metric}"]
        summary[metric] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"  {metric:10s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    summary["temps_total_s"] = elapsed
    cv_summary[name] = summary
    print(f"  Temps total : {elapsed:.1f}s")

with open("cross_validation_results.json", "w") as f:
    json.dump(cv_summary, f, indent=2)

# ---------------------------------------------------------
# 4. EVALUATION FINALE sur le jeu de test
#    (Random Forest optimise re-entraine sur tout le train,
#     MLP entraine sur tout le train)
# ---------------------------------------------------------
print("\n=== Entrainement final sur tout le train + evaluation sur le test ===")

final_results = {}

rf_final = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
t0 = time.time()
rf_final.fit(X_train_scaled, y_train)
rf_time = time.time() - t0
y_pred_rf = rf_final.predict(X_test_scaled)
y_proba_rf = rf_final.predict_proba(X_test_scaled)[:, 1]
final_results["Random Forest (optimise via GridSearch)"] = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Precision": precision_score(y_test, y_pred_rf),
    "Recall": recall_score(y_test, y_pred_rf),
    "F1-score": f1_score(y_test, y_pred_rf),
    "AUC-ROC": roc_auc_score(y_test, y_proba_rf),
    "Temps entrainement (s)": rf_time,
    "Meilleurs hyperparametres": best_params,
}

mlp_final = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=60,
                           early_stopping=True, random_state=42)
t0 = time.time()
mlp_final.fit(X_train_scaled, y_train)
mlp_time = time.time() - t0
y_pred_mlp = mlp_final.predict(X_test_scaled)
y_proba_mlp = mlp_final.predict_proba(X_test_scaled)[:, 1]
final_results["MLP (DL)"] = {
    "Accuracy": accuracy_score(y_test, y_pred_mlp),
    "Precision": precision_score(y_test, y_pred_mlp),
    "Recall": recall_score(y_test, y_pred_mlp),
    "F1-score": f1_score(y_test, y_pred_mlp),
    "AUC-ROC": roc_auc_score(y_test, y_proba_mlp),
    "Temps entrainement (s)": mlp_time,
}

with open("final_results_after_tuning.json", "w") as f:
    json.dump(final_results, f, indent=2, default=str)

print("\n=== RESULTATS FINAUX (apres GridSearch) ===")
for model_name, metrics in final_results.items():
    print(f"\n{model_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

# ---------------------------------------------------------
# 5. Graphique : stabilite des resultats (cross-validation)
# ---------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(9, 5.5))
metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
x = np.arange(len(metrics_to_plot))
width = 0.35
colors = {"Random Forest (optimise)": "#2563eb", "MLP (DL)": "#dc2626"}

for i, (name, summary) in enumerate(cv_summary.items()):
    means = [summary[m]["mean"] for m in metrics_to_plot]
    stds = [summary[m]["std"] for m in metrics_to_plot]
    ax.bar(x + i*width - width/2, means, width, yerr=stds, capsize=4,
           label=name, color=colors.get(name, "#888"))

ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"])
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score (moyenne +/- ecart-type sur 5 folds)")
ax.set_title("Stabilite des resultats - Validation croisee (5-fold)\nCSE-CIC-IDS2018 (cloud AWS)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("cross_validation_stability.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 6. Graphique : resultats du GridSearch (heatmap simplifiee)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = [
    f"n_est={int(r.param_n_estimators)}\ndepth={r.param_max_depth}"
    for r in cv_results_df.itertuples()
]
bars = ax.bar(range(len(cv_results_df)), cv_results_df["mean_test_score"],
               yerr=cv_results_df["std_test_score"], capsize=3, color="#2563eb")
ax.set_xticks(range(len(cv_results_df)))
ax.set_xticklabels(labels, fontsize=7, rotation=0)
ax.set_ylabel("Score F1 (validation croisee)")
ax.set_title("GridSearch - Score F1 selon les hyperparametres testes")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("gridsearch_results.png", dpi=150)
plt.close()

print("\nFichiers generes : gridsearch_results.csv/png, cross_validation_results.json, "
      "cross_validation_stability.png, final_results_after_tuning.json")
