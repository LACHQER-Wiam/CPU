# Random Forest — Parallélisation sur CPU

Ce projet s'inscrit dans le cadre du cours **programmation parallèle sur CPU** 

L'objectif est d'implémenter un algorithme de Random Forest en plusieurs versions, du plus naïf au plus optimisé, afin de mesurer et comprendre le gain apporté par chaque technique de parallélisation et d'optimisation bas niveau étudiée en cours.

---

## Contenu du projet

```
.
├── naive.py                      # Version naïve (Python pur)
├── optimised.py                  # Version parallélisée (joblib / concurrent.futures)
├── rf_cython.pyx                 # Fonctions critiques compilées en C (Cython)
├── setup.py                      # Script de compilation du module Cython
├── optimised_cython.py           # Random Forest utilisant le module Cython
├── benchmark_random_forest.ipynb # Notebook de comparaison des versions
├── requirements.txt              # Dépendances Python du projet
└── README.md                     # Ce fichier
```

### `naive.py`

Implémentation **séquentielle en Python pur** d'une Random Forest. Aucune parallélisation, aucune dépendance à NumPy : les données sont stockées dans des listes Python classiques. Ce fichier sert de référence de base pour mesurer les gains des versions optimisées. Il illustre le goulot d'étranglement de l'interpréteur Python sur des boucles intensives (calcul du Gini, recherche de coupure).

Classe exposée : `RandomForestNaive`

### `optimised.py`

Implémentation **parallélisée sur CPU** utilisant les deux approches vues en cours :

- `RandomForestJoblib` : parallélisation via `joblib.Parallel` avec le backend `loky` (pool de processus persistant). Chaque arbre est entraîné dans un processus indépendant, ce qui contourne le GIL Python. Les données sont stockées en NumPy pour de meilleurs performances mémoire.
- `RandomForestFutures` : même principe avec `concurrent.futures.ProcessPoolExecutor`, une API plus bas niveau qui permet de comprendre le mécanisme de soumission et de collecte des tâches.

Les deux classes appliquent la stratégie de **tâches grossières** (un arbre entier par tâche) pour amortir le coût de communication entre processus (sérialisation pickle).

### `rf_cython.pyx`

Fichier source **Cython** contenant les deux fonctions les plus coûteuses de l'algorithme, réécrites en C typé :

- `gini_cython` : calcul de l'impureté de Gini avec des variables `cdef` et des memoryviews NumPy, sans passer par l'interpréteur Python.
- `meilleure_coupure_cython` : boucle de recherche du meilleur seuil de coupure, compilée en C. C'est ici que se concentre l'essentiel du temps de calcul.

Ce fichier doit être **compilé** avant utilisation avec la commande décrite ci-dessous.

### `setup.py`

Script de compilation du module Cython. Il utilise `setuptools` et `Cython.Build.cythonize` pour transformer `rf_cython.pyx` en un fichier `.so` importable directement par Python. Les directives `boundscheck=False` et `wraparound=False` sont activées pour maximiser la vitesse du code compilé.

### `optimised_cython.py`

Random Forest qui **appelle le module Cython compilé** pour les fonctions critiques. Deux classes sont disponibles :

- `RandomForestCython` : version séquentielle avec les fonctions Cython. Permet d'isoler le gain apporté par la compilation C, indépendamment du parallélisme.
- `RandomForestCythonParallel` : combine Cython et joblib — la meilleure version du projet. Chaque arbre est construit dans un processus séparé, et chaque processus utilise les fonctions C compilées pour les calculs.

### `benchmark_random_forest.ipynb`

Notebook Jupyter de comparaison des cinq versions sur le dataset **Heart Disease UCI** (303 lignes, 13 features, classification binaire). Il contient :

- La compilation automatique du module Cython au démarrage
- Le chargement et la préparation du dataset
- Un benchmark avec variation du nombre d'arbres (5, 10, 20, 40) sur plusieurs runs
- Un tableau de speedups par rapport à la version naïve
- Des graphiques : courbes de temps, barres de speedup, boxplots de variabilité

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
python -m venv env
```

**2. Activer l'environnement**

Sur Linux / macOS :
```bash
source env/bin/activate
```

Sur Windows :
```bash
env\Scripts\activate
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
jupyter notebook benchmark_random_forest.ipynb
```


### Désactiver l'environnement

```bash
deactivate
```

---
