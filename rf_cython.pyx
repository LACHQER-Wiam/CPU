# rf_cython.pyx
# -------------
# Fonctions Cython compilées en C pour la Random Forest
#
# Ce fichier contient 4 fonctions critiques :
#   - gini_cython              : impureté de Gini (classification)
#   - meilleure_coupure_classif: meilleur seuil de coupure (classification)
#   - mse_cython               : MSE d'un nœud (régression)
#   - meilleure_coupure_regress: meilleur seuil de coupure (régression)


import numpy as np
cimport numpy as np


# ===========================================================================
# CLASSIFICATION — Impureté de Gini
# ===========================================================================

def gini_cython(np.int64_t[:] y):
    """
    Calcule l'impureté de Gini d'un tableau d'étiquettes entières.
    Gini = 1 - sum(p_k^2)
    Retourne 0.0 pour un nœud pur, ~0.5 pour un nœud impur
    """
    cdef int n = y.shape[0]
    cdef int i
    cdef int max_classe = 0
    cdef double p
    cdef double somme = 0.0

    if n == 0:
        return 0.0

    # Trouver la classe maximale pour dimensionner le tableau de comptage
    for i in range(n):
        if y[i] > max_classe:
            max_classe = y[i]

    # Comptage des occurrences de chaque classe
    counts = np.zeros(max_classe + 1, dtype=np.int64)
    cdef np.int64_t[:] counts_view = counts
    for i in range(n):
        counts_view[y[i]] += 1

    # Gini = 1 - sum(p_k^2)
    for i in range(max_classe + 1):
        p = <double>counts_view[i] / n
        somme += p * p

    return 1.0 - somme


# ===========================================================================
# CLASSIFICATION — Meilleure coupure (Gini)
# ===========================================================================

def meilleure_coupure_classif(
    np.float64_t[:] feature_vals,
    np.int64_t[:]   y,
    double          gini_parent
):
    """
    Cherche le meilleur seuil de coupure sur UNE feature pour la classification.

    Paramètres :
        feature_vals : valeurs de la feature pour tous les exemples
        y            : étiquettes entières
        gini_parent  : Gini du nœud parent (déjà calculé, évite de le recalculer)

    Retourne : (meilleur_seuil, meilleur_gain)
    """
    cdef int n = feature_vals.shape[0]
    cdef int i, ig, id_
    cdef int idx, n_seuils
    cdef double seuil, gain
    cdef double meilleur_gain  = -1.0
    cdef double meilleur_seuil = 0.0
    cdef int n_gauche, n_droite
    cdef double gini_g, gini_d
    cdef np.int64_t[:] yg
    cdef np.int64_t[:] yd

    seuils_array = np.unique(np.asarray(feature_vals))
    cdef np.float64_t[:] seuils = seuils_array
    n_seuils = seuils.shape[0]

    for idx in range(n_seuils):
        seuil = seuils[idx]

        # Compter les exemples à gauche / droite
        n_gauche = 0
        n_droite = 0
        for i in range(n):
            if feature_vals[i] <= seuil:
                n_gauche += 1
            else:
                n_droite += 1

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

        gini_g = gini_cython(yg)
        gini_d = gini_cython(yd)

        gain = gini_parent - (
            (<double>n_gauche / n) * gini_g +
            (<double>n_droite / n) * gini_d
        )

        if gain > meilleur_gain:
            meilleur_gain  = gain
            meilleur_seuil = seuil

    return meilleur_seuil, meilleur_gain


# ===========================================================================
# RÉGRESSION — MSE d'un nœud
# ===========================================================================

def mse_cython(np.float64_t[:] y):
    """
    Calcule la MSE (variance) d'un tableau de valeurs réelles.
    MSE = mean((y_i - mean(y))^2)
    Retourne 0.0 si toutes les valeurs sont identiques.
    """
    cdef int n = y.shape[0]
    cdef int i
    cdef double somme = 0.0
    cdef double moy
    cdef double mse = 0.0

    if n == 0:
        return 0.0

    # Calcul de la moyenne
    for i in range(n):
        somme += y[i]
    moy = somme / n

    # Calcul de la MSE
    for i in range(n):
        mse += (y[i] - moy) * (y[i] - moy)

    return mse / n


# ===========================================================================
# RÉGRESSION — Meilleure coupure (réduction de MSE)
# ===========================================================================

def meilleure_coupure_regress(
    np.float64_t[:] feature_vals,
    np.float64_t[:] y,
    double          mse_parent
):
    """
    Cherche le meilleur seuil de coupure sur UNE feature pour la régression.

    Paramètres :
        feature_vals : valeurs de la feature pour tous les exemples
        y            : valeurs cibles réelles
        mse_parent   : MSE du nœud parent (déjà calculée)

    Retourne : (meilleur_seuil, meilleur_gain)
    """
    cdef int n = feature_vals.shape[0]
    cdef int i, ig, id_
    cdef int idx, n_seuils
    cdef double seuil, gain
    cdef double meilleur_gain  = -1.0
    cdef double meilleur_seuil = 0.0
    cdef int n_gauche, n_droite
    cdef double mse_g, mse_d
    cdef np.float64_t[:] yg
    cdef np.float64_t[:] yd

    seuils_array = np.unique(np.asarray(feature_vals))
    cdef np.float64_t[:] seuils = seuils_array
    n_seuils = seuils.shape[0]

    for idx in range(n_seuils):
        seuil = seuils[idx]

        n_gauche = 0
        n_droite = 0
        for i in range(n):
            if feature_vals[i] <= seuil:
                n_gauche += 1
            else:
                n_droite += 1

        if n_gauche == 0 or n_droite == 0:
            continue

        # Construire y_gauche et y_droite (float64 pour la régression)
        y_gauche = np.empty(n_gauche, dtype=np.float64)
        y_droite = np.empty(n_droite, dtype=np.float64)
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

        mse_g = mse_cython(yg)
        mse_d = mse_cython(yd)

        # Gain = réduction de MSE
        gain = mse_parent - (
            (<double>n_gauche / n) * mse_g +
            (<double>n_droite / n) * mse_d
        )

        if gain > meilleur_gain:
            meilleur_gain  = gain
            meilleur_seuil = seuil

    return meilleur_seuil, meilleur_gain
