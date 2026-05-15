"""
naive.py
--------
Implémentation NAÏVE d'une Random Forest entièrement en Python pur.
Aucune parallélisation : les arbres sont construits un par un, séquentiellement.

Concepts du cours illustrés :
  - Pas d'exploitation des cœurs disponibles (un seul thread actif)
  - Coût O(n_arbres × n_samples × n_features × profondeur)
  - Point de référence pour mesurer le gain apporté par la parallélisation
"""

import math
import random
import time
from collections import Counter


# ---------------------------------------------------------------------------
# Nœud d'un arbre de décision
# ---------------------------------------------------------------------------

class Noeud:
    """Représente un nœud interne ou une feuille d'un arbre de décision."""

    def __init__(self):
        self.est_feuille = False
        self.classe = None          # Valeur prédite si feuille
        self.feature_idx = None     # Indice de la feature utilisée pour la coupure
        self.seuil = None           # Valeur seuil de la coupure
        self.gauche = None          # Sous-arbre gauche  (feature <= seuil)
        self.droite = None          # Sous-arbre droit   (feature >  seuil)


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def _impurete_gini(y):
    """Calcule l'impureté de Gini d'un vecteur d'étiquettes (liste Python)."""
    n = len(y)
    if n == 0:
        return 0.0
    compteur = Counter(y)
    # Gini = 1 - sum(p_k^2)
    gini = 1.0 - sum((c / n) ** 2 for c in compteur.values())
    return gini


def _meilleure_coupure(X, y, indices_features):
    """
    Cherche la meilleure coupure parmi un sous-ensemble de features.
    Parcours naïf : on teste tous les seuils possibles pour chaque feature.

    Retourne (feature_idx, seuil) ou (None, None) si aucune coupure utile.
    """
    n = len(y)
    meilleur_gain = -1.0
    meilleur_feat = None
    meilleur_seuil = None
    gini_parent = _impurete_gini(y)

    for feat_idx in indices_features:
        # Récupération des valeurs de la feature pour tous les exemples
        valeurs = [X[i][feat_idx] for i in range(n)]
        seuils_candidats = sorted(set(valeurs))

        for seuil in seuils_candidats:
            y_gauche = [y[i] for i in range(n) if X[i][feat_idx] <= seuil]
            y_droite = [y[i] for i in range(n) if X[i][feat_idx] > seuil]

            if len(y_gauche) == 0 or len(y_droite) == 0:
                continue

            # Gain d'information = Gini parent - moyenne pondérée des enfants
            gain = gini_parent - (
                (len(y_gauche) / n) * _impurete_gini(y_gauche)
                + (len(y_droite) / n) * _impurete_gini(y_droite)
            )

            if gain > meilleur_gain:
                meilleur_gain = gain
                meilleur_feat = feat_idx
                meilleur_seuil = seuil

    return meilleur_feat, meilleur_seuil


def _construire_arbre(X, y, profondeur_max, n_features_par_split, profondeur=0):
    """
    Construit récursivement un arbre de décision.

    Paramètres
    ----------
    X                  : liste de listes (n_samples × n_features)
    y                  : liste d'étiquettes (n_samples,)
    profondeur_max     : profondeur maximale de l'arbre
    n_features_par_split : nombre de features tirées aléatoirement à chaque nœud
    profondeur         : profondeur courante (pour la récursion)
    """
    noeud = Noeud()
    classe_majoritaire = Counter(y).most_common(1)[0][0]

    # Critères d'arrêt
    if (profondeur >= profondeur_max
            or len(set(y)) == 1
            or len(y) <= 1):
        noeud.est_feuille = True
        noeud.classe = classe_majoritaire
        return noeud

    # Tirage aléatoire d'un sous-ensemble de features (Random Forest)
    n_features_total = len(X[0])
    k = min(n_features_par_split, n_features_total)
    indices_features = random.sample(range(n_features_total), k)

    feat_idx, seuil = _meilleure_coupure(X, y, indices_features)

    # Aucune coupure utile → feuille
    if feat_idx is None:
        noeud.est_feuille = True
        noeud.classe = classe_majoritaire
        return noeud

    # Partition des données
    X_g = [X[i] for i in range(len(y)) if X[i][feat_idx] <= seuil]
    y_g = [y[i] for i in range(len(y)) if X[i][feat_idx] <= seuil]
    X_d = [X[i] for i in range(len(y)) if X[i][feat_idx] > seuil]
    y_d = [y[i] for i in range(len(y)) if X[i][feat_idx] > seuil]

    noeud.feature_idx = feat_idx
    noeud.seuil = seuil

    # Récursion sur les deux branches
    noeud.gauche = _construire_arbre(X_g, y_g, profondeur_max,
                                      n_features_par_split, profondeur + 1)
    noeud.droite = _construire_arbre(X_d, y_d, profondeur_max,
                                      n_features_par_split, profondeur + 1)
    return noeud


def _predire_un_exemple(noeud, x):
    """Descend dans l'arbre pour prédire la classe d'un exemple x."""
    if noeud.est_feuille:
        return noeud.classe
    if x[noeud.feature_idx] <= noeud.seuil:
        return _predire_un_exemple(noeud.gauche, x)
    else:
        return _predire_un_exemple(noeud.droite, x)


# ---------------------------------------------------------------------------
# Random Forest naïve (séquentielle)
# ---------------------------------------------------------------------------

class RandomForestNaive:
    """
    Random Forest NAÏVE : les n_arbres sont entraînés les uns après les autres
    dans une simple boucle for. Aucun parallélisme.

    Méthodes du cours non appliquées :
      - Pas de joblib / concurrent.futures
      - Pas de partage mémoire optimisé
      - Structures de données Python pures (listes), pas NumPy
    """

    def __init__(self, n_arbres=10, profondeur_max=5, n_features_par_split=None,
                 graine=42):
        self.n_arbres = n_arbres
        self.profondeur_max = profondeur_max
        self.n_features_par_split = n_features_par_split  # None → sqrt(n_features)
        self.graine = graine
        self.arbres_ = []  # Liste des arbres entraînés

    def fit(self, X, y):
        """
        Entraîne la forêt sur (X, y).

        X : liste de listes (n_samples × n_features)
        y : liste d'étiquettes
        """
        random.seed(self.graine)
        n_samples = len(y)
        n_features = len(X[0])

        k = self.n_features_par_split or max(1, int(math.sqrt(n_features)))
        self.arbres_ = []

        # Boucle SÉQUENTIELLE : c'est ici le goulot d'étranglement
        for _ in range(self.n_arbres):
            # Bootstrap : tirage avec remise de n_samples indices
            indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]

            arbre = _construire_arbre(X_boot, y_boot, self.profondeur_max, k)
            self.arbres_.append(arbre)

        return self

    def predict(self, X):
        """
        Prédit la classe pour chaque exemple de X par vote majoritaire.
        Retourne une liste de prédictions.
        """
        predictions = []
        for x in X:
            votes = [_predire_un_exemple(arbre, x) for arbre in self.arbres_]
            classe = Counter(votes).most_common(1)[0][0]
            predictions.append(classe)
        return predictions

    def score(self, X, y):
        """Retourne la précision (accuracy) sur (X, y)."""
        preds = self.predict(X)
        correct = sum(p == vrai for p, vrai in zip(preds, y))
        return correct / len(y)


# ---------------------------------------------------------------------------
# Point d'entrée rapide pour tester le module seul
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Génération d'un petit jeu de données synthétique
    random.seed(0)
    n = 200
    X_train = [[random.gauss(0, 1) for _ in range(6)] for _ in range(n)]
    y_train = [1 if x[0] + x[1] > 0 else 0 for x in X_train]

    print("=== Random Forest Naïve ===")
    debut = time.perf_counter()
    rf = RandomForestNaive(n_arbres=20, profondeur_max=5)
    rf.fit(X_train, y_train)
    duree = time.perf_counter() - debut

    acc = rf.score(X_train, y_train)
    print(f"  Temps d'entraînement : {duree:.3f} s")
    print(f"  Accuracy sur train   : {acc:.3f}")
