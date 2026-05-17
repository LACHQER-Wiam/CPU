"""
optimised_cython.py
--------------------
Random Forest avec les fonctions critiques compilées en C (Cython).

Deux classes sont disponibles :
  - RandomForestClassifieurCython  : classification (Gini compilé en C)
  - RandomForestRegresseurCython   : régression     (MSE compilée en C)

Prérequis : avoir compilé rf_cython.pyx avec
    python setup.py build_ext --inplace

Différence avec naive.py :
  - Les boucles de calcul Gini/MSE et la recherche de seuil tournent en C.
  - Les données sont stockées en tableaux NumPy (mémoire contiguë).
  - La structure de l'arbre (récursion, bootstrap) reste en Python.
"""

import math
import time
import numpy as np
from collections import Counter

# Import du module Cython compilé
import rf_cython


# ===========================================================================
# Structure commune : un nœud d'arbre
# ===========================================================================

class Noeud:
    """Nœud d'un arbre de décision (partagé entre classification et régression)."""
    def __init__(self):
        self.est_feuille = False
        self.valeur      = None   # classe (classif) ou moyenne (regress)
        self.feature_idx = None
        self.seuil       = None
        self.gauche      = None
        self.droite      = None


# ===========================================================================
# Fonction de prédiction commune
# ===========================================================================

def _predire_un_exemple(noeud, x):
    """Descend dans l'arbre et retourne la valeur de la feuille."""
    if noeud.est_feuille:
        return noeud.valeur
    if x[noeud.feature_idx] <= noeud.seuil:
        return _predire_un_exemple(noeud.gauche, x)
    else:
        return _predire_un_exemple(noeud.droite, x)


# ===========================================================================
# CLASSIFICATION — construction de l'arbre (utilise Cython pour Gini + seuil)
# ===========================================================================

def _construire_arbre_classif(X, y, profondeur_max, k, profondeur=0):
    """
    Construit un arbre de décision pour la classification.
    Appelle rf_cython.gini_cython et rf_cython.meilleure_coupure_classif.

    X : tableau NumPy float64 (n_samples × n_features)
    y : tableau NumPy int64   (n_samples,)
    """
    noeud = Noeud()

    # Classe majoritaire = valeur de la feuille
    classes, counts = np.unique(y, return_counts=True)
    noeud.valeur = int(classes[np.argmax(counts)])

    # Critères d'arrêt
    if profondeur >= profondeur_max or len(classes) == 1 or len(y) <= 1:
        noeud.est_feuille = True
        return noeud

    n_features = X.shape[1]
    indices_features = np.random.choice(n_features, min(k, n_features), replace=False)

    # Appel Cython : Gini du nœud parent
    gini_parent = rf_cython.gini_cython(y)

    meilleur_feat  = None
    meilleur_seuil = None
    meilleur_gain  = -1.0

    for feat_idx in indices_features:
        # np.ascontiguousarray garantit un tableau contigu en mémoire (requis par Cython)
        feature_vals = np.ascontiguousarray(X[:, feat_idx], dtype=np.float64)

        # Appel Cython : meilleur seuil pour cette feature
        seuil, gain = rf_cython.meilleure_coupure_classif(feature_vals, y, gini_parent)

        if gain > meilleur_gain:
            meilleur_gain  = gain
            meilleur_feat  = feat_idx
            meilleur_seuil = seuil

    if meilleur_feat is None or meilleur_gain <= 0:
        noeud.est_feuille = True
        return noeud

    masque_g = X[:, meilleur_feat] <= meilleur_seuil
    noeud.feature_idx = int(meilleur_feat)
    noeud.seuil       = meilleur_seuil
    noeud.gauche  = _construire_arbre_classif(
        X[masque_g],  y[masque_g],  profondeur_max, k, profondeur + 1)
    noeud.droite  = _construire_arbre_classif(
        X[~masque_g], y[~masque_g], profondeur_max, k, profondeur + 1)
    return noeud


# ===========================================================================
# RÉGRESSION — construction de l'arbre (utilise Cython pour MSE + seuil)
# ===========================================================================

def _construire_arbre_regress(X, y, profondeur_max, k, profondeur=0):
    """
    Construit un arbre de décision pour la régression.
    Appelle rf_cython.mse_cython et rf_cython.meilleure_coupure_regress.

    X : tableau NumPy float64 (n_samples × n_features)
    y : tableau NumPy float64 (n_samples,)
    """
    noeud = Noeud()
    noeud.valeur = float(np.mean(y))

    # Critères d'arrêt
    if profondeur >= profondeur_max or len(y) <= 1 or np.std(y) < 1e-10:
        noeud.est_feuille = True
        return noeud

    n_features = X.shape[1]
    indices_features = np.random.choice(n_features, min(k, n_features), replace=False)

    # Appel Cython : MSE du nœud parent
    mse_parent = rf_cython.mse_cython(y)

    meilleur_feat  = None
    meilleur_seuil = None
    meilleur_gain  = -1.0

    for feat_idx in indices_features:
        feature_vals = np.ascontiguousarray(X[:, feat_idx], dtype=np.float64)

        # Appel Cython : meilleur seuil pour cette feature
        seuil, gain = rf_cython.meilleure_coupure_regress(feature_vals, y, mse_parent)

        if gain > meilleur_gain:
            meilleur_gain  = gain
            meilleur_feat  = feat_idx
            meilleur_seuil = seuil

    if meilleur_feat is None or meilleur_gain <= 0:
        noeud.est_feuille = True
        return noeud

    masque_g = X[:, meilleur_feat] <= meilleur_seuil
    noeud.feature_idx = int(meilleur_feat)
    noeud.seuil       = meilleur_seuil
    noeud.gauche  = _construire_arbre_regress(
        X[masque_g],  y[masque_g],  profondeur_max, k, profondeur + 1)
    noeud.droite  = _construire_arbre_regress(
        X[~masque_g], y[~masque_g], profondeur_max, k, profondeur + 1)
    return noeud


# ===========================================================================
# Classe publique — Classification Cython
# ===========================================================================

class RandomForestClassifieurCython:
    """
    Random Forest de CLASSIFICATION avec fonctions Cython.

    Critère de coupure : Gini compilé en C (rf_cython.gini_cython).
    Agrégation          : vote majoritaire entre les T arbres.
    Structures          : tableaux NumPy (mémoire contiguë).
    Parallélisation     : aucune — on isole le gain Cython pur.
    """

    def __init__(self, n_arbres=10, profondeur_max=5,
                 n_features_par_split=None, graine=42):
        self.n_arbres            = n_arbres
        self.profondeur_max      = profondeur_max
        self.n_features_par_split = n_features_par_split
        self.graine              = graine
        self.arbres_             = []

    def fit(self, X, y):
        """Entraîne la forêt. X : NumPy float64, y : NumPy int64."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int64)
        np.random.seed(self.graine)

        n, p = X.shape
        k = self.n_features_par_split or max(1, int(math.sqrt(p)))
        self.arbres_ = []

        for _ in range(self.n_arbres):
            idx    = np.random.randint(0, n, size=n)
            X_boot = np.ascontiguousarray(X[idx])
            y_boot = np.ascontiguousarray(y[idx])
            self.arbres_.append(
                _construire_arbre_classif(X_boot, y_boot, self.profondeur_max, k)
            )
        return self

    def predict(self, X):
        """Vote majoritaire entre les arbres."""
        X = np.asarray(X, dtype=np.float64)
        predictions = []
        for x in X:
            votes = [_predire_un_exemple(a, x) for a in self.arbres_]
            predictions.append(Counter(votes).most_common(1)[0][0])
        return predictions

    def score(self, X, y):
        """Accuracy."""
        preds = self.predict(X)
        return np.mean(np.array(preds) == np.array(y))


# ===========================================================================
# Classe publique — Régression Cython
# ===========================================================================

class RandomForestRegresseurCython:
    """
    Random Forest de RÉGRESSION avec fonctions Cython.

    Critère de coupure : réduction de MSE compilée en C (rf_cython.mse_cython).
    Agrégation          : moyenne des prédictions des T arbres.
    Structures          : tableaux NumPy (mémoire contiguë).
    Parallélisation     : aucune — on isole le gain Cython pur.
    """

    def __init__(self, n_arbres=10, profondeur_max=5,
                 n_features_par_split=None, graine=42):
        self.n_arbres            = n_arbres
        self.profondeur_max      = profondeur_max
        self.n_features_par_split = n_features_par_split
        self.graine              = graine
        self.arbres_             = []

    def fit(self, X, y):
        """Entraîne la forêt. X : NumPy float64, y : NumPy float64."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        np.random.seed(self.graine)

        n, p = X.shape
        k = self.n_features_par_split or max(1, p // 3)
        self.arbres_ = []

        for _ in range(self.n_arbres):
            idx    = np.random.randint(0, n, size=n)
            X_boot = np.ascontiguousarray(X[idx])
            y_boot = np.ascontiguousarray(y[idx])
            self.arbres_.append(
                _construire_arbre_regress(X_boot, y_boot, self.profondeur_max, k)
            )
        return self

    def predict(self, X):
        """Moyenne des prédictions de chaque arbre."""
        X = np.asarray(X, dtype=np.float64)
        predictions = []
        for x in X:
            votes = [_predire_un_exemple(a, x) for a in self.arbres_]
            predictions.append(float(np.mean(votes)))
        return predictions

    def score(self, X, y):
        """R² score."""
        preds = np.array(self.predict(X))
        y     = np.asarray(y, dtype=np.float64)
        ss_res = np.sum((preds - y) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ===========================================================================
# Test rapide
# ===========================================================================

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n   = 200

    # --- Classification ---
    X   = rng.randn(n, 6).astype(np.float64)
    y_c = ((X[:, 0] + X[:, 1]) > 0).astype(np.int64)

    print("=== Classification Cython ===")
    t0 = time.perf_counter()
    clf = RandomForestClassifieurCython(n_arbres=15, profondeur_max=5)
    clf.fit(X, y_c)
    print(f"  Temps : {time.perf_counter()-t0:.3f}s | Accuracy : {clf.score(X, y_c):.3f}")

    # --- Régression ---
    y_r = (2 * X[:, 0] + X[:, 1] + rng.randn(n) * 0.5).astype(np.float64)

    print("=== Régression Cython ===")
    t0 = time.perf_counter()
    reg = RandomForestRegresseurCython(n_arbres=15, profondeur_max=5)
    reg.fit(X, y_r)
    print(f"  Temps : {time.perf_counter()-t0:.3f}s | R²     : {reg.score(X, y_r):.3f}")
