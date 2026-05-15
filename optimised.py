"""
optimised.py
------------
Implémentation PARALLÉLISÉE d'une Random Forest sur CPU.

Méthodes du cours appliquées (Séances 3 & 4) :
  1. Parallélisation par PROCESSUS avec joblib (évite le GIL Python)
  2. Utilisation de concurrent.futures.ProcessPoolExecutor comme alternative
  3. Bootstrap + construction de chaque arbre dans un processus indépendant
  4. Agrégation des résultats (réduction = vote majoritaire)
  5. NumPy pour les opérations sur tableaux (meilleur usage du cache)

Rappel du cours :
  - Le GIL Python empêche le vrai parallélisme des THREADS en Python pur
    → on utilise des PROCESSUS (chaque processus a son propre GIL)
  - Coût de communication : sérialisation (pickle) des données entre processus
  - Stratégie : tâches grossières (un arbre complet par tâche) pour amortir
    le coût de démarrage des processus
  - joblib.Parallel avec backend="loky" utilise un pool de processus persistant
    → moins de surcoût que de relancer des processus à chaque appel
"""

import math
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Nœud d'un arbre (identique à naive.py, mais les données sont NumPy)
# ---------------------------------------------------------------------------

class Noeud:
    """Représente un nœud interne ou une feuille d'un arbre de décision."""

    def __init__(self):
        self.est_feuille = False
        self.classe = None
        self.feature_idx = None
        self.seuil = None
        self.gauche = None
        self.droite = None


# ---------------------------------------------------------------------------
# Fonctions de construction d'arbre (versions NumPy, plus rapides)
# ---------------------------------------------------------------------------

def _impurete_gini_np(y):
    """Calcule l'impureté de Gini avec NumPy (vectorisé, meilleur usage du cache L1/L2)."""
    n = len(y)
    if n == 0:
        return 0.0
    # np.unique compte les classes efficacement
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - np.sum(probs ** 2)


def _meilleure_coupure_np(X, y, indices_features):
    """
    Cherche la meilleure coupure en utilisant NumPy pour le slicing et les masques.
    Plus efficace en cache qu'une liste Python car les données sont contiguës en mémoire.
    """
    n = len(y)
    meilleur_gain = -1.0
    meilleur_feat = None
    meilleur_seuil = None
    gini_parent = _impurete_gini_np(y)

    for feat_idx in indices_features:
        valeurs = X[:, feat_idx]
        seuils_candidats = np.unique(valeurs)

        for seuil in seuils_candidats:
            masque_g = valeurs <= seuil
            masque_d = ~masque_g

            if masque_g.sum() == 0 or masque_d.sum() == 0:
                continue

            y_g = y[masque_g]
            y_d = y[masque_d]

            gain = gini_parent - (
                (len(y_g) / n) * _impurete_gini_np(y_g)
                + (len(y_d) / n) * _impurete_gini_np(y_d)
            )

            if gain > meilleur_gain:
                meilleur_gain = gain
                meilleur_feat = feat_idx
                meilleur_seuil = seuil

    return meilleur_feat, meilleur_seuil


def _construire_arbre_np(X, y, profondeur_max, n_features_par_split, profondeur=0):
    """
    Construction récursive de l'arbre avec NumPy.
    Appelée dans chaque processus fils → pas de contention (pas de GIL partagé).
    """
    noeud = Noeud()
    classes, counts = np.unique(y, return_counts=True)
    classe_majoritaire = classes[np.argmax(counts)]

    # Critères d'arrêt
    if profondeur >= profondeur_max or len(classes) == 1 or len(y) <= 1:
        noeud.est_feuille = True
        noeud.classe = classe_majoritaire
        return noeud

    n_features = X.shape[1]
    k = min(n_features_par_split, n_features)
    indices_features = np.random.choice(n_features, k, replace=False)

    feat_idx, seuil = _meilleure_coupure_np(X, y, indices_features)

    if feat_idx is None:
        noeud.est_feuille = True
        noeud.classe = classe_majoritaire
        return noeud

    masque_g = X[:, feat_idx] <= seuil
    noeud.feature_idx = feat_idx
    noeud.seuil = seuil
    noeud.gauche = _construire_arbre_np(
        X[masque_g], y[masque_g], profondeur_max, n_features_par_split, profondeur + 1
    )
    noeud.droite = _construire_arbre_np(
        X[~masque_g], y[~masque_g], profondeur_max, n_features_par_split, profondeur + 1
    )
    return noeud


def _predire_un_exemple_np(noeud, x):
    """Descend dans l'arbre pour prédire la classe de l'exemple x (vecteur NumPy)."""
    if noeud.est_feuille:
        return noeud.classe
    if x[noeud.feature_idx] <= noeud.seuil:
        return _predire_un_exemple_np(noeud.gauche, x)
    else:
        return _predire_un_exemple_np(noeud.droite, x)


# ---------------------------------------------------------------------------
# Fonction TOP-LEVEL pour la parallélisation par processus
# ---------------------------------------------------------------------------
# IMPORTANT : la fonction doit être au niveau du module (top-level) pour être
# sérialisable par pickle, condition nécessaire pour multiprocessing.

def _entrainer_un_arbre(X, y, profondeur_max, n_features_par_split, graine):
    """
    Entraîne UN seul arbre sur un bootstrap de (X, y).
    Cette fonction est exécutée dans un processus fils.

    La graine est passée explicitement pour garantir la reproductibilité
    même en parallèle (chaque processus a son propre générateur de nombres aléatoires).
    """
    # Initialisation du générateur aléatoire LOCAL au processus
    np.random.seed(graine)
    random.seed(graine)

    n_samples = X.shape[0]
    # Bootstrap : tirage avec remise
    indices = np.random.randint(0, n_samples, size=n_samples)
    X_boot = X[indices]
    y_boot = y[indices]

    arbre = _construire_arbre_np(X_boot, y_boot, profondeur_max, n_features_par_split)
    return arbre


# ---------------------------------------------------------------------------
# Random Forest parallélisée (joblib)
# ---------------------------------------------------------------------------

class RandomForestJoblib:
    """
    Random Forest parallélisée avec joblib.Parallel (backend "loky").

    joblib.Parallel lance un pool de processus persistants (Loky).
    Avantage : pas de surcoût de démarrage à chaque appel.
    Chaque tâche = entraînement d'un arbre complet (tâche "grossière").

    Concept du cours : tâches grossières pour amortir le coût de communication
    (sérialisation pickle des données X, y entre processus).
    """

    def __init__(self, n_arbres=10, profondeur_max=5, n_features_par_split=None,
                 n_jobs=-1, graine=42):
        self.n_arbres = n_arbres
        self.profondeur_max = profondeur_max
        self.n_features_par_split = n_features_par_split
        self.n_jobs = n_jobs      # -1 = utiliser tous les cœurs disponibles
        self.graine = graine
        self.arbres_ = []

    def fit(self, X, y):
        """
        Entraîne la forêt en parallèle avec joblib.

        X : tableau NumPy (n_samples × n_features)
        y : tableau NumPy (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_features = X.shape[1]
        k = self.n_features_par_split or max(1, int(math.sqrt(n_features)))

        # Génération de graines reproductibles pour chaque arbre
        rng = np.random.RandomState(self.graine)
        graines = rng.randint(0, 10_000_000, size=self.n_arbres)

        # Parallélisation : chaque arbre est entraîné dans un processus séparé
        # n_jobs=-1 → autant de processus que de cœurs CPU disponibles
        self.arbres_ = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_entrainer_un_arbre)(X, y, self.profondeur_max, k, int(g))
            for g in graines
        )
        return self

    def predict(self, X):
        """Prédiction par vote majoritaire (séquentiel, la prédiction est rapide)."""
        X = np.asarray(X, dtype=np.float64)
        predictions = []
        for x in X:
            votes = [_predire_un_exemple_np(arbre, x) for arbre in self.arbres_]
            classe = Counter(votes).most_common(1)[0][0]
            predictions.append(classe)
        return predictions

    def score(self, X, y):
        """Retourne la précision (accuracy)."""
        preds = self.predict(X)
        y = np.asarray(y)
        return np.mean(np.array(preds) == y)


# ---------------------------------------------------------------------------
# Random Forest parallélisée (concurrent.futures) — alternative plus bas niveau
# ---------------------------------------------------------------------------

class RandomForestFutures:
    """
    Random Forest parallélisée avec concurrent.futures.ProcessPoolExecutor.

    Version plus bas niveau que joblib, utile pour comprendre le mécanisme :
      1. On crée un pool de processus
      2. On soumet les tâches (futures)
      3. On collecte les résultats au fur et à mesure (as_completed)

    Équivalent fonctionnel à RandomForestJoblib, mais avec plus de contrôle.
    Illustre le concept du cours "Parallelization with processes (concurrent.futures)".
    """

    def __init__(self, n_arbres=10, profondeur_max=5, n_features_par_split=None,
                 n_workers=None, graine=42):
        self.n_arbres = n_arbres
        self.profondeur_max = profondeur_max
        self.n_features_par_split = n_features_par_split
        self.n_workers = n_workers   # None → os.cpu_count()
        self.graine = graine
        self.arbres_ = []

    def fit(self, X, y):
        """Entraîne la forêt avec ProcessPoolExecutor."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_features = X.shape[1]
        k = self.n_features_par_split or max(1, int(math.sqrt(n_features)))

        rng = np.random.RandomState(self.graine)
        graines = rng.randint(0, 10_000_000, size=self.n_arbres)

        arbres = [None] * self.n_arbres

        # Soumission des tâches au pool de processus
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # map des futures → index pour reconstruire l'ordre
            futures = {
                executor.submit(
                    _entrainer_un_arbre, X, y, self.profondeur_max, k, int(g)
                ): i
                for i, g in enumerate(graines)
            }
            # Collecte des résultats dès qu'ils sont disponibles
            for future in as_completed(futures):
                idx = futures[future]
                arbres[idx] = future.result()

        self.arbres_ = arbres
        return self

    def predict(self, X):
        """Prédiction par vote majoritaire."""
        X = np.asarray(X, dtype=np.float64)
        predictions = []
        for x in X:
            votes = [_predire_un_exemple_np(arbre, x) for arbre in self.arbres_]
            classe = Counter(votes).most_common(1)[0][0]
            predictions.append(classe)
        return predictions

    def score(self, X, y):
        """Retourne la précision (accuracy)."""
        preds = self.predict(X)
        y = np.asarray(y)
        return np.mean(np.array(preds) == y)


# ---------------------------------------------------------------------------
# Point d'entrée rapide pour tester le module seul
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n = 200
    X_train = rng.randn(n, 6)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    print("=== Random Forest Parallélisée (joblib) ===")
    debut = time.perf_counter()
    rf = RandomForestJoblib(n_arbres=20, profondeur_max=5, n_jobs=-1)
    rf.fit(X_train, y_train)
    duree = time.perf_counter() - debut
    acc = rf.score(X_train, y_train)
    print(f"  Temps d'entraînement : {duree:.3f} s")
    print(f"  Accuracy sur train   : {acc:.3f}")

    print("\n=== Random Forest Parallélisée (concurrent.futures) ===")
    debut = time.perf_counter()
    rf2 = RandomForestFutures(n_arbres=20, profondeur_max=5)
    rf2.fit(X_train, y_train)
    duree2 = time.perf_counter() - debut
    acc2 = rf2.score(X_train, y_train)
    print(f"  Temps d'entraînement : {duree2:.3f} s")
    print(f"  Accuracy sur train   : {acc2:.3f}")
