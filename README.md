# Random Forest — Parallélisation sur CPU

Ce projet s'inscrit dans le cadre du cours **programmation parallèle sur CPU** 

L'objectif est d'implémenter un algorithme de Random Forest en plusieurs versions, du plus naïf au plus optimisé, afin de mesurer et comprendre le gain apporté par chaque technique de parallélisation et d'optimisation bas niveau étudiée en cours.

---

## Structure du projet

```
.
├── naive.py                  # Version naïve (Python pur, séquentiel)
├── rf_cython.pyx             # Fonctions critiques compilées en C (Cython)
├── rf_cython.c               # Code C généré automatiquement par Cython (ne pas modifier)
├── setup.py                  # Script de compilation du module Cython
├── optimised_cython.py       # Random Forest utilisant Cython + parallélisation joblib
├── benchmark_complet.ipynb   # Notebook de comparaison des versions
├── requirements.txt          # Dépendances Python du projet
└── README.md                 # Ce fichier
```

---

## Description des fichiers et fonctions

### `naive.py`

Implémentation **séquentielle en Python pur** d'une Random Forest. Aucune dépendance à NumPy : les données circulent sous forme de listes Python classiques. Sert de baseline pour mesurer les gains des versions optimisées.

**Fonctions internes (classification)**

| Fonction | Rôle |
|---|---|
| `_gini(y)` | Calcule l'impureté de Gini d'une liste d'étiquettes : `1 - Σ pₖ²` |
| `_meilleure_coupure_classif(X, y, indices_features)` | Parcourt tous les seuils candidats sur les features tirées et retourne la coupure qui maximise le gain de Gini |
| `_construire_arbre_classif(X, y, profondeur_max, k)` | Construit récursivement un arbre de décision par partitionnement Gini |

**Fonctions internes (régression)**

| Fonction | Rôle |
|---|---|
| `_mse(y)` | Calcule la MSE (variance) d'une liste de valeurs réelles |
| `_meilleure_coupure_regress(X, y, indices_features)` | Même principe que la classification, mais minimise la MSE pondérée des enfants |
| `_construire_arbre_regress(X, y, profondeur_max, k)` | Construit récursivement un arbre de régression |

**Fonction partagée**

| Fonction | Rôle |
|---|---|
| `_predire_un_exemple(noeud, x)` | Descend un exemple dans l'arbre en suivant les règles de coupure jusqu'à une feuille |

**Classes exposées**

| Classe | Tâche | Agrégation | Features par nœud |
|---|---|---|---|
| `RandomForestClassifieurNaif` | Classification | Vote majoritaire | `sqrt(p)` |
| `RandomForestRegresseurNaif` | Régression | Moyenne | `p / 3` |

---

### `rf_cython.pyx`

Réécriture en **C typé** des 4 fonctions qui concentrent l'essentiel du temps de calcul. Utilise des `cdef`, des memoryviews NumPy (`np.float64_t[:]`, `np.int64_t[:]`) et des variables locales typées pour éliminer le passage par l'interpréteur Python.

| Fonction | Rôle |
|---|---|
| `gini_cython(y)` | Calcul de l'impureté de Gini sur un memoryview `int64` |
| `meilleure_coupure_classif(feature_vals, y, gini_parent)` | Recherche du meilleur seuil sur une feature pour la classification |
| `mse_cython(y)` | Calcul de la MSE sur un memoryview `float64` |
| `meilleure_coupure_regress(feature_vals, y, mse_parent)` | Recherche du meilleur seuil sur une feature pour la régression |

> Ce fichier doit être **compilé** avant utilisation (voir section Installation).

---

### `setup.py`

Script de compilation Cython. Transforme `rf_cython.pyx` → `rf_cython.c` → `rf_cython.so` (`.pyd` sur Windows) via `setuptools` et `Cython.Build.cythonize`. Les directives `boundscheck=False` et `wraparound=False` désactivent les vérifications de bornes pour maximiser la vitesse.

---

### `optimised_cython.py`

Random Forest qui **appelle les fonctions Cython compilées** pour les calculs critiques, et **parallélise la construction des arbres** avec `joblib.Parallel` (backend `loky`, processus séparés — contourne le GIL).

**Fonctions internes**

| Fonction | Rôle |
|---|---|
| `_construire_arbre_classif(X, y, profondeur_max, k)` | Construction d'un arbre de classification avec appels Cython pour Gini et coupure |
| `_construire_arbre_regress(X, y, profondeur_max, k)` | Construction d'un arbre de régression avec appels Cython pour MSE et coupure |
| `_build_tree_classif(X_boot, y_boot, profondeur_max, k, seed)` | Wrapper top-level (picklable) appelé par joblib pour la classification |
| `_build_tree_regress(X_boot, y_boot, profondeur_max, k, seed)` | Wrapper top-level (picklable) appelé par joblib pour la régression |
| `_predire_un_exemple(noeud, x)` | Descente dans l'arbre, identique à la version naïve |

**Classes exposées**

| Classe | Tâche | Agrégation | Parallélisation |
|---|---|---|---|
| `RandomForestClassifieurCython` | Classification | Vote majoritaire | `joblib.Parallel(n_jobs=-1)` |
| `RandomForestRegresseurCython` | Régression | Moyenne | `joblib.Parallel(n_jobs=-1)` |

Paramètre `n_jobs=-1` par défaut : utilise tous les cœurs disponibles.

---

### `benchmark_complet.ipynb`

Notebook Jupyter de comparaison des **3 versions** (Naïve, Cython+joblib, Sklearn) sur **4 datasets**, avec variation du nombre d'arbres.

| Cellule | Contenu |
|---|---|
| 1 | Compilation automatique du module Cython |
| 2 | Imports + paramètres (`N_RUNS=10`, `TREES_LIST=[5, 15, 25]`, `DEPTH=5`) |
| 3 | Chargement des 4 datasets |
| 4 | Fonction `run_benchmark` + définition des datasets |
| 5 | Exécution du benchmark (boucle sur les 3 valeurs de `TREES_LIST`) |
| 6 | Histogrammes des temps moyens — grille 3 × 4 (lignes = nb arbres, colonnes = datasets) |
| 7 | Tableau récapitulatif (temps, std, accuracy/R², speedup vs Naïve) |
| 8 | Graphique de speedup — Cython+joblib et Sklearn vs Naïve |

**Datasets utilisés**

| Dataset | Tâche | Taille |
|---|---|---|
| Moons (synthétique) | Classification | 600 points, 2 features |
| Breast Cancer | Classification | 569 patients, 30 features |
| Sinusoïde bruitée (synthétique) | Régression | 400 points, 1 feature |
| California Housing | Régression | 2 000 maisons, 8 features |

---

## Créer l'environnement avec venv

### Prérequis

- Python 3.10 ou supérieur
- Un compilateur C installé sur la machine :
  - **Linux** : `sudo apt install build-essential`
  - **macOS** : `xcode-select --install`
  - **Windows** : installer Visual Studio Build Tools

### Étapes

**1. Créer l'environnement virtuel**

```bash
python -m venv venv
```

**2. Activer l'environnement**

Sur Linux / macOS :
```bash
source venv/bin/activate
```

Sur Windows :
```bash
venv\Scripts\activate
```


**3. Installer les dépendances**

```bash
pip install -r requirements.txt
```

**4. Compiler le module Cython**

Cette étape génère le fichier `rf_cython.so` (ou `.pyd` sur Windows) nécessaire à `optimised_cython.py` :

```bash
python setup.py build_ext --inplace
```

**5. Lancer le notebook**

```bash
jupyter notebook benchmark_complet.ipynb
```


### Désactiver l'environnement

```bash
deactivate
```

---
