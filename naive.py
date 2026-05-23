"""
naive.py
--------
Implémentation naïve d'une Random Forest séquentielle 

Deux classes sont disponibles :
  - RandomForestClassifieurNaif  : classification (vote majoritaire, critère Gini)
  - RandomForestRegresseurNaif   : régression     (moyenne des feuilles, critère MSE)

Aucune parallélisation, aucune dépendance NumPy :
tout repose sur des listes Python et des boucles for
C'est le point de référence pour mesurer le gain de la version Cython
"""

import math
import random
import time
from collections import Counter



### Structure commune : un nœud d'arbre


class Noeud:
    """
    Nœud d'un arbre de décision.
    Utilisé à la fois pour la classification et la régression.
    """
    def __init__(self):
        self.est_feuille    = False
        self.valeur         = None   # classe majoritaire (classif) ou moyenne (regress)
        self.feature_idx    = None   # indice de la feature de coupure
        self.seuil          = None   # valeur du seuil de coupure
        self.gauche         = None   # sous-arbre gauche  (feature <= seuil)
        self.droite         = None   # sous-arbre droit   (feature >  seuil)



### Fonctions partagées


def _predire_un_exemple(noeud, x):
    """Fait descendre l'exemple x dans l'arbre et retourne la valeur de la feuille."""
    if noeud.est_feuille:
        return noeud.valeur
    if x[noeud.feature_idx] <= noeud.seuil:
        return _predire_un_exemple(noeud.gauche, x)
    else:
        return _predire_un_exemple(noeud.droite, x)



### CLASSIFICATION — fonctions internes


def _gini(y):
    """
    Impureté de Gini d'une liste d'étiquettes y 
    Gini = 1 - sum(p_k^2)  = 0 si nœud pur, ~0.5 si nœud impur
    """
    n = len(y)
    if n == 0:
        return 0.0
    compteur = Counter(y)
    return 1.0 - sum((c / n) ** 2 for c in compteur.values())


def _meilleure_coupure_classif(X, y, indices_features):
    """
    Parcourt tous les seuils possibles pour chaque feature tirée,
    et retourne la coupure qui maximise le gain de Gini
    """
    n = len(y)
    meilleur_gain  = -1.0
    meilleur_feat  = None
    meilleur_seuil = None
    gini_parent    = _gini(y)

    for feat_idx in indices_features:
        valeurs           = [X[i][feat_idx] for i in range(n)]
        seuils_candidats  = sorted(set(valeurs))

        for seuil in seuils_candidats:
            y_g = [y[i] for i in range(n) if X[i][feat_idx] <= seuil]
            y_d = [y[i] for i in range(n) if X[i][feat_idx] >  seuil]

            if not y_g or not y_d:
                continue

            gain = gini_parent - (
                (len(y_g) / n) * _gini(y_g) +
                (len(y_d) / n) * _gini(y_d)
            )
            if gain > meilleur_gain:
                meilleur_gain  = gain
                meilleur_feat  = feat_idx
                meilleur_seuil = seuil

    return meilleur_feat, meilleur_seuil


def _construire_arbre_classif(X, y, profondeur_max, k, profondeur=0):
    """Construction récursive d'un arbre de décision pour la classification."""
    noeud = Noeud()
    # Valeur de la feuille = classe la plus fréquente
    noeud.valeur = Counter(y).most_common(1)[0][0]

    # Critères d'arrêt
    if profondeur >= profondeur_max or len(set(y)) == 1 or len(y) <= 1:
        noeud.est_feuille = True
        return noeud

    # Tirage aléatoire d'un sous-ensemble de features
    n_features = len(X[0])
    indices_features = random.sample(range(n_features), min(k, n_features))

    feat_idx, seuil = _meilleure_coupure_classif(X, y, indices_features)

    if feat_idx is None:
        noeud.est_feuille = True
        return noeud

    # Partition gauche / droite
    X_g = [X[i] for i in range(len(y)) if X[i][feat_idx] <= seuil]
    y_g = [y[i] for i in range(len(y)) if X[i][feat_idx] <= seuil]
    X_d = [X[i] for i in range(len(y)) if X[i][feat_idx] >  seuil]
    y_d = [y[i] for i in range(len(y)) if X[i][feat_idx] >  seuil]

    noeud.feature_idx = feat_idx
    noeud.seuil       = seuil
    noeud.gauche      = _construire_arbre_classif(X_g, y_g, profondeur_max, k, profondeur + 1)
    noeud.droite      = _construire_arbre_classif(X_d, y_d, profondeur_max, k, profondeur + 1)
    return noeud



### RÉGRESSION — fonctions internes


def _mse(y):
    """
    Erreur quadratique moyenne (MSE) autour de la moyenne.
    MSE = 0 si tous les y sont identiques.
    """
    n = len(y)
    if n == 0:
        return 0.0
    moy = sum(y) / n
    return sum((v - moy) ** 2 for v in y) / n


def _meilleure_coupure_regress(X, y, indices_features):
    """
    Cherche la coupure qui minimise la MSE pondérée des deux enfants.
    Critère : réduction de variance (MSE parent - MSE pondérée enfants).
    """
    n = len(y)
    meilleur_gain  = -1.0
    meilleur_feat  = None
    meilleur_seuil = None
    mse_parent     = _mse(y)

    for feat_idx in indices_features:
        valeurs          = [X[i][feat_idx] for i in range(n)]
        seuils_candidats = sorted(set(valeurs))

        for seuil in seuils_candidats:
            y_g = [y[i] for i in range(n) if X[i][feat_idx] <= seuil]
            y_d = [y[i] for i in range(n) if X[i][feat_idx] >  seuil]

            if not y_g or not y_d:
                continue

            # Gain = réduction de MSE
            gain = mse_parent - (
                (len(y_g) / n) * _mse(y_g) +
                (len(y_d) / n) * _mse(y_d)
            )
            if gain > meilleur_gain:
                meilleur_gain  = gain
                meilleur_feat  = feat_idx
                meilleur_seuil = seuil

    return meilleur_feat, meilleur_seuil


def _construire_arbre_regress(X, y, profondeur_max, k, profondeur=0):
    """Construction récursive d'un arbre de décision pour la régression."""
    noeud = Noeud()
    # Valeur de la feuille = moyenne des y
    noeud.valeur = sum(y) / len(y)

    # Critères d'arrêt
    if profondeur >= profondeur_max or len(set(y)) == 1 or len(y) <= 1:
        noeud.est_feuille = True
        return noeud

    n_features       = len(X[0])
    indices_features = random.sample(range(n_features), min(k, n_features))

    feat_idx, seuil = _meilleure_coupure_regress(X, y, indices_features)

    if feat_idx is None:
        noeud.est_feuille = True
        return noeud

    X_g = [X[i] for i in range(len(y)) if X[i][feat_idx] <= seuil]
    y_g = [y[i] for i in range(len(y)) if X[i][feat_idx] <= seuil]
    X_d = [X[i] for i in range(len(y)) if X[i][feat_idx] >  seuil]
    y_d = [y[i] for i in range(len(y)) if X[i][feat_idx] >  seuil]

    noeud.feature_idx = feat_idx
    noeud.seuil       = seuil
    noeud.gauche      = _construire_arbre_regress(X_g, y_g, profondeur_max, k, profondeur + 1)
    noeud.droite      = _construire_arbre_regress(X_d, y_d, profondeur_max, k, profondeur + 1)
    return noeud



### Classe publique — Classification naïve


class RandomForestClassifieurNaif:
    """
    Random Forest de classification 

    Critère de coupure : impureté de Gini
    Agrégation          : vote majoritaire entre les T arbres
    Structures          : listes Python
    Parallélisation     : aucune (boucle for séquentielle)
    """

    def __init__(self, n_arbres=10, profondeur_max=5,
                 n_features_par_split=None, graine=42):
        self.n_arbres            = n_arbres
        self.profondeur_max      = profondeur_max
        self.n_features_par_split = n_features_par_split  # None → sqrt(p)
        self.graine              = graine
        self.arbres_             = []

    def fit(self, X, y):
        """
        Entraîne la forêt sur X (liste de listes) et y (liste d'entiers)
        Chaque arbre est construit sur un bootstrap de X, y
        """
        random.seed(self.graine)
        n       = len(y)
        p       = len(X[0])
        k       = self.n_features_par_split or max(1, int(math.sqrt(p)))
        self.arbres_ = []

        for _ in range(self.n_arbres):
            # Bootstrap : tirage avec remise
            idx    = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in idx]
            y_boot = [y[i] for i in idx]
            self.arbres_.append(
                _construire_arbre_classif(X_boot, y_boot, self.profondeur_max, k)
            )
        return self

    def predict(self, X):
        """Retourne la classe prédite pour chaque exemple"""
        predictions = []
        for x in X:
            votes = [_predire_un_exemple(a, x) for a in self.arbres_]
            predictions.append(Counter(votes).most_common(1)[0][0])
        return predictions

    def score(self, X, y):
        """Accuracy : proportion de prédictions correctes."""
        preds = self.predict(X)
        return sum(p == v for p, v in zip(preds, y)) / len(y)



### Classe publique — Régression naïve


class RandomForestRegresseurNaif:
    """
    Random Forest de RÉGRESSION en Python pur.

    Critère de coupure : réduction de MSE (variance).
    Agrégation          : moyenne des prédictions des T arbres.
    Structures          : listes Python (pas NumPy).
    Parallélisation     : aucune (boucle for séquentielle).
    """

    def __init__(self, n_arbres=10, profondeur_max=5,
                 n_features_par_split=None, graine=42):
        self.n_arbres            = n_arbres
        self.profondeur_max      = profondeur_max
        self.n_features_par_split = n_features_par_split  # None → p/3 
        self.graine              = graine
        self.arbres_             = []

    def fit(self, X, y):
        """
        Entraîne la forêt sur X  et y 
        """
        random.seed(self.graine)
        n       = len(y)
        p       = len(X[0])
        # Convention régression : p/3 features par nœud 
        k       = self.n_features_par_split or max(1, p // 3)
        self.arbres_ = []

        for _ in range(self.n_arbres):
            idx    = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in idx]
            y_boot = [y[i] for i in idx]
            self.arbres_.append(
                _construire_arbre_regress(X_boot, y_boot, self.profondeur_max, k)
            )
        return self

    def predict(self, X):
        """Retourne la moyenne des prédictions de chaque arbre."""
        predictions = []
        for x in X:
            votes = [_predire_un_exemple(a, x) for a in self.arbres_]
            predictions.append(sum(votes) / len(votes))
        return predictions

    def score(self, X, y):
        """R² score : 1 − MSE / Var(y)."""
        preds = self.predict(X)
        y_moy = sum(y) / len(y)
        ss_res = sum((p - v) ** 2 for p, v in zip(preds, y))
        ss_tot = sum((v - y_moy) ** 2 for v in y)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

