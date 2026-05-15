"""
optimised_cython.py
--------------------
Random Forest utilisant les fonctions Cython compilées (rf_cython.pyx).

Architecture simple :
  - rf_cython.pyx  : les 2 fonctions critiques compilées en C (Gini + coupure)
  - ce fichier     : la structure de l'arbre et la forêt restent en Python

Pourquoi séparer ainsi ?
  Le cours montre qu'on n'a pas besoin de tout réécrire en Cython.
  On identifie le GOULOT D'ÉTRANGLEMENT (la boucle de recherche de coupure)
  et on ne compile que ça. C'est le principe du profiling : optimiser
  là où le temps est vraiment passé.

Prérequis :
  Avoir compilé rf_cython.pyx avec :
      python setup.py build_ext --inplace
"""

import math
import random
import time
from collections import Counter

import numpy as np

# Import du module Cython compilé
# Si cette ligne échoue, il faut d'abord compiler :
#   python setup.py build_ext --inplace
import rf_cython


# ---------------------------------------------------------------------------
# Structure d'un nœud (identique aux autres versions)
# ---------------------------------------------------------------------------

class Noeud:
    """Un nœud d'arbre de décision — feuille ou nœud interne."""

    def __init__(self):
        self.est_feuille = False
        self.classe = None        # Classe prédite (si feuille)
        self.feature_idx = None   # Quelle feature on coupe
        self.seuil = None         # Valeur du seuil de coupure
        self.gauche = None        # Sous-arbre gauche  (feature <= seuil)
        self.droite = None        # Sous-arbre droit   (feature >  seuil)


# ---------------------------------------------------------------------------
# Construction de l'arbre — utilise les fonctions Cython pour la coupure
# ---------------------------------------------------------------------------

def _construire_arbre_cython(X, y, profondeur_max, n_features_par_split, profondeur=0):
    """
    Construit un arbre de décision récursivement.

    La différence avec naive.py :
      - On appelle rf_cython.gini_cython() au lieu de Counter()
      - On appelle rf_cython.meilleure_coupure_cython() au lieu de la boucle Python

    X : tableau NumPy (n_samples × n_features), float64
    y : tableau NumPy (n_samples,), int64
    """
    noeud = Noeud()

    # Classe majoritaire (pour les feuilles)
    classes, counts = np.unique(y, return_counts=True)
    classe_majoritaire = classes[np.argmax(counts)]

    # Critères d'arrêt
    if profondeur >= profondeur_max or len(classes) == 1 or len(y) <= 1:
        noeud.est_feuille = True
        noeud.classe = classe_majoritaire
        return noeud

    # Tirage aléatoire d'un sous-ensemble de features (Random Forest)
    n_features = X.shape[1]
    k = min(n_features_par_split, n_features)
    indices_features = np.random.choice(n_features, k, replace=False)

    # ---- APPEL CYTHON : calcul du Gini du nœud parent ----
    gini_parent = rf_cython.gini_cython(y)

    # Chercher la meilleure coupure parmi les features tirées
    meilleur_feat = None
    meilleur_seuil = None
    meilleur_gain = -1.0

    for feat_idx in indices_features:
        # On extrait les valeurs de cette feature (contiguous pour Cython)
        feature_vals = np.ascontiguousarray(X[:, feat_idx], dtype=np.float64)

        # ---- APPEL CYTHON : recherche du meilleur seuil ----
        seuil, gain = rf_cython.meilleure_coupure_cython(
            feature_vals, y, gini_parent
        )

        if gain > meilleur_gain:
            meilleur_gain = gain
            meilleur_feat = feat_idx
            meilleur_seuil = seuil

    # Aucune coupure utile → feuille
    if meilleur_feat is None or meilleur_gain <= 0:
        noeud.est_feuille = True
        noeud.classe = classe_majoritaire
        return noeud

    # Partitionner les données selon la meilleure coupure
    masque_g = X[:, meilleur_feat] <= meilleur_seuil
    noeud.feature_idx = meilleur_feat
    noeud.seuil = meilleur_seuil

    # Récursion sur les deux branches
    noeud.gauche = _construire_arbre_cython(
        X[masque_g],  y[masque_g],  profondeur_max, n_features_par_split, profondeur + 1
    )
    noeud.droite = _construire_arbre_cython(
        X[~masque_g], y[~masque_g], profondeur_max, n_features_par_split, profondeur + 1
    )
    return noeud


def _predire_exemple(noeud, x):
    """Descend dans l'arbre pour prédire la classe de x."""
    if noeud.est_feuille:
        return noeud.classe
    if x[noeud.feature_idx] <= noeud.seuil:
        return _predire_exemple(noeud.gauche, x)
    else:
        return _predire_exemple(noeud.droite, x)


# ---------------------------------------------------------------------------
# Fonction top-level pour la parallélisation (doit être au niveau du module)
# ---------------------------------------------------------------------------

def _entrainer_un_arbre_cython(X, y, profondeur_max, n_features_par_split, graine):
    """
    Entraîne un seul arbre avec bootstrap.
    Appelée dans un processus fils par joblib.
    Utilise les fonctions Cython pour la partie calcul.
    """
    np.random.seed(graine)
    n = X.shape[0]
    indices = np.random.randint(0, n, size=n)
    X_boot = np.ascontiguousarray(X[indices], dtype=np.float64)
    y_boot = np.ascontiguousarray(y[indices], dtype=np.int64)
    return _construire_arbre_cython(X_boot, y_boot, profondeur_max, n_features_par_split)


# ---------------------------------------------------------------------------
# Random Forest Cython (séquentielle — pour comparer avec naive.py)
# ---------------------------------------------------------------------------

class RandomForestCython:
    """
    Random Forest utilisant les fonctions Cython pour les calculs critiques.
    Version SÉQUENTIELLE : on compare avec naive.py pour isoler le gain Cython.

    Gain attendu vs naive.py :
      - Gini calculé en C (sans Counter Python)
      - Boucle de recherche de seuil en C (sans list comprehension Python)
    """

    def __init__(self, n_arbres=10, profondeur_max=5, n_features_par_split=None, graine=42):
        self.n_arbres = n_arbres
        self.profondeur_max = profondeur_max
        self.n_features_par_split = n_features_par_split
        self.graine = graine
        self.arbres_ = []

    def fit(self, X, y):
        """Entraîne la forêt séquentiellement avec les fonctions Cython."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int64)
        n_features = X.shape[1]
        k = self.n_features_par_split or max(1, int(math.sqrt(n_features)))

        np.random.seed(self.graine)
        self.arbres_ = []

        for i in range(self.n_arbres):
            # Bootstrap
            n = X.shape[0]
            indices = np.random.randint(0, n, size=n)
            X_boot = np.ascontiguousarray(X[indices])
            y_boot = np.ascontiguousarray(y[indices])

            # Construction avec fonctions Cython
            arbre = _construire_arbre_cython(X_boot, y_boot, self.profondeur_max, k)
            self.arbres_.append(arbre)

        return self

    def predict(self, X):
        """Prédiction par vote majoritaire."""
        X = np.asarray(X, dtype=np.float64)
        predictions = []
        for x in X:
            votes = [_predire_exemple(arbre, x) for arbre in self.arbres_]
            predictions.append(Counter(votes).most_common(1)[0][0])
        return predictions

    def score(self, X, y):
        """Retourne l'accuracy."""
        preds = self.predict(X)
        return np.mean(np.array(preds) == np.array(y))


# ---------------------------------------------------------------------------
# Random Forest Cython + Parallèle (la version la plus optimisée)
# ---------------------------------------------------------------------------

class RandomForestCythonParallel:
    """
    Combinaison des deux optimisations du cours :
      1. Cython  : calculs critiques compilés en C
      2. joblib  : arbres construits en parallèle sur N cœurs

    C'est la version la plus proche de sklearn en termes d'architecture.
    """

    def __init__(self, n_arbres=10, profondeur_max=5, n_features_par_split=None,
                 n_jobs=-1, graine=42):
        self.n_arbres = n_arbres
        self.profondeur_max = profondeur_max
        self.n_features_par_split = n_features_par_split
        self.n_jobs = n_jobs
        self.graine = graine
        self.arbres_ = []

    def fit(self, X, y):
        """Entraîne la forêt en parallèle avec joblib + Cython."""
        from joblib import Parallel, delayed

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int64)
        n_features = X.shape[1]
        k = self.n_features_par_split or max(1, int(math.sqrt(n_features)))

        rng = np.random.RandomState(self.graine)
        graines = rng.randint(0, 10_000_000, size=self.n_arbres)

        # Parallélisation : chaque arbre dans un processus, avec Cython
        self.arbres_ = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_entrainer_un_arbre_cython)(X, y, self.profondeur_max, k, int(g))
            for g in graines
        )
        return self

    def predict(self, X):
        """Prédiction par vote majoritaire."""
        X = np.asarray(X, dtype=np.float64)
        predictions = []
        for x in X:
            votes = [_predire_exemple(arbre, x) for arbre in self.arbres_]
            predictions.append(Counter(votes).most_common(1)[0][0])
        return predictions

    def score(self, X, y):
        """Retourne l'accuracy."""
        preds = self.predict(X)
        return np.mean(np.array(preds) == np.array(y))


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n = 300
    X = rng.randn(n, 6).astype(np.float64)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.int64)

    print("=== Random Forest Cython (séquentielle) ===")
    debut = time.perf_counter()
    rf = RandomForestCython(n_arbres=20, profondeur_max=5)
    rf.fit(X, y)
    print(f"  Temps : {time.perf_counter() - debut:.3f}s | Accuracy : {rf.score(X, y):.3f}")

    print("\n=== Random Forest Cython + Parallèle ===")
    debut = time.perf_counter()
    rf2 = RandomForestCythonParallel(n_arbres=20, profondeur_max=5, n_jobs=-1)
    rf2.fit(X, y)
    print(f"  Temps : {time.perf_counter() - debut:.3f}s | Accuracy : {rf2.score(X, y):.3f}")
