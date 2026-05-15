# rf_cython.pyx
# -------------
# Version Cython des fonctions CRITIQUES de la Random Forest.
#
# Pourquoi Cython ?
#   En Python pur, chaque operation passe par l'interpreteur (overhead).
#   Cython compile ce fichier en C, ce qui supprime cet overhead.
#
# Ce qu'on fait ici :
#   1. On declare les TYPES des variables avec "cdef int", "cdef double"
#      -> Cython genere du C type, beaucoup plus rapide
#   2. On utilise des vues memoire (memoryview) sur les tableaux NumPy
#      -> acces direct aux donnees en memoire, sans copie
#
# REGLE CYTHON IMPORTANTE :
#   Tous les "cdef" doivent etre declares EN HAUT de la fonction,
#   avant tout autre code. C'est une contrainte du C sous-jacent.

import numpy as np
cimport numpy as np


# ------------------------------------------------------------------
# FONCTION 1 : calcul de l'impurete de Gini
# ------------------------------------------------------------------
def gini_cython(np.int64_t[:] y):
    """
    Calcule l'impurete de Gini d'un tableau d'etiquettes entieres.
    Retourne un float entre 0.0 (noeud pur) et ~0.5 (noeud impur).
    """
    # --- Toutes les declarations C en haut de la fonction ---
    cdef int n = y.shape[0]
    cdef int i
    cdef int max_classe = 0
    cdef double p
    cdef double somme = 0.0

    if n == 0:
        return 0.0

    # Trouver la valeur max pour dimensionner le tableau de comptage
    for i in range(n):
        if y[i] > max_classe:
            max_classe = y[i]

    # Tableau de comptage (une case par classe)
    counts = np.zeros(max_classe + 1, dtype=np.int64)
    cdef np.int64_t[:] counts_view = counts

    # Compter les occurrences de chaque classe
    for i in range(n):
        counts_view[y[i]] += 1

    # Gini = 1 - somme(p_k^2)
    for i in range(max_classe + 1):
        p = <double>counts_view[i] / n
        somme += p * p

    return 1.0 - somme


# ------------------------------------------------------------------
# FONCTION 2 : meilleure coupure sur UNE feature
# ------------------------------------------------------------------
def meilleure_coupure_cython(
    np.float64_t[:] feature_vals,
    np.int64_t[:] y,
    double gini_parent
):
    """
    Cherche le meilleur seuil de coupure pour une feature donnee.

    Parametres :
        feature_vals : valeurs de la feature (1 colonne du dataset)
        y            : etiquettes (classes)
        gini_parent  : impurete du noeud avant la coupure

    Retourne :
        (meilleur_seuil, meilleur_gain)
        Si aucune coupure utile : (0.0, -1.0)
    """
    # --- Toutes les declarations C en haut ---
    cdef int n = feature_vals.shape[0]
    cdef int i, ig, id_
    cdef int idx, n_seuils
    cdef double seuil
    cdef double gain
    cdef double meilleur_gain = -1.0
    cdef double meilleur_seuil = 0.0
    cdef int n_gauche, n_droite
    cdef double gini_g, gini_d
    cdef np.int64_t[:] yg
    cdef np.int64_t[:] yd

    # Seuils candidats = valeurs uniques de la feature
    seuils_array = np.unique(np.asarray(feature_vals))
    cdef np.float64_t[:] seuils = seuils_array
    n_seuils = seuils.shape[0]

    # Boucle principale : on teste chaque seuil possible
    for idx in range(n_seuils):
        seuil = seuils[idx]

        # Compter les exemples a gauche et a droite
        n_gauche = 0
        n_droite = 0
        for i in range(n):
            if feature_vals[i] <= seuil:
                n_gauche += 1
            else:
                n_droite += 1

        # Coupure inutile si un cote est vide
        if n_gauche == 0 or n_droite == 0:
            continue

        # Construire y_gauche et y_droite
        y_gauche = np.empty(n_gauche, dtype=np.int64)
        y_droite = np.empty(n_droite, dtype=np.int64)
        yg = y_gauche
        yd = y_droite

        ig = 0
        id_ = 0
        for i in range(n):
            if feature_vals[i] <= seuil:
                yg[ig] = y[i]
                ig += 1
            else:
                yd[id_] = y[i]
                id_ += 1

        # Calcul du gain d'information
        gini_g = gini_cython(yg)
        gini_d = gini_cython(yd)

        gain = gini_parent - (
            (<double>n_gauche / n) * gini_g +
            (<double>n_droite / n) * gini_d
        )

        if gain > meilleur_gain:
            meilleur_gain = gain
            meilleur_seuil = seuil

    return meilleur_seuil, meilleur_gain
