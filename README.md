# Random Forest — Parallélisation sur CPU

Ce projet s'inscrit dans le cadre du cours **programmation parallèle sur CPU** 

L'objectif est d'implémenter un algorithme de Random Forest en plusieurs versions, du plus naïf au plus optimisé, afin de mesurer et comprendre le gain apporté par chaque technique de parallélisation et d'optimisation bas niveau étudiée en cours.

---

## Contenu du projet

```
.
├── naive.py                  # Version naïve (Python pur)
├── rf_cython.pyx             # Fonctions critiques compilées en C (Cython)
├── setup.py                  # Script de compilation du module Cython
├── optimised_cython.py       # Random Forest utilisant le module Cython
├── benchmark_complet.ipynb   # Notebook de comparaison des versions
├── requirements.txt          # Dépendances Python du projet
└── README.md                 # Ce fichier
```

### `naive.py`

Implémentation **séquentielle en Python pur** d'une Random Forest pour la classification et la régression. Aucune parallélisation, aucune dépendance à NumPy : les données sont stockées dans des listes Python classiques. Ce fichier sert de référence de base pour mesurer les gains des versions optimisées. Il illustre le goulot d'étranglement de l'interpréteur Python sur des boucles intensives (calcul du Gini, recherche de coupure).

Classes exposées : `RandomForestClassifieurNaif`, `RandomForestRegresseurNaif`

### `rf_cython.pyx`

Fichier source **Cython** contenant les deux fonctions les plus coûteuses de l'algorithme, réécrites en C typé :

- `gini_cython` : calcul de l'impureté de Gini avec des variables `cdef` et des memoryviews NumPy, sans passer par l'interpréteur Python.
- `meilleure_coupure_cython` : boucle de recherche du meilleur seuil de coupure, compilée en C. C'est ici que se concentre l'essentiel du temps de calcul.

Ce fichier doit être **compilé** avant utilisation avec la commande décrite ci-dessous.

### `setup.py`

Script de compilation du module Cython. Il utilise `setuptools` et `Cython.Build.cythonize` pour transformer `rf_cython.pyx` en un fichier `.so` importable directement par Python. Les directives `boundscheck=False` et `wraparound=False` sont activées pour maximiser la vitesse du code compilé.

### `optimised_cython.py`

Random Forest qui **appelle le module Cython compilé** pour les fonctions critiques, couvrant les tâches de classification et de régression :

- `RandomForestClassifieurCython` : classifieur avec les fonctions Cython. Permet d'isoler le gain apporté par la compilation C, indépendamment du parallélisme.
- `RandomForestRegresseurCython` : régresseur avec les fonctions Cython.

### `benchmark_complet.ipynb`

Notebook Jupyter de comparaison des **3 versions** (Naïve, Cython, Sklearn) sur **4 datasets** couvrant classification et régression. Il contient :

- La compilation automatique du module Cython au démarrage
- La visualisation des 4 datasets
- Un benchmark sur 5 runs (20 arbres, profondeur 5) pour chaque dataset :
  - **Classification** : Moons (synthétique, 600 points) et Breast Cancer (569 patients, 30 features)
  - **Régression** : Sinusoïde bruitée (synthétique, 400 points) et California Housing (2 000 maisons, 8 features)
- Des graphiques détaillés : frontières de décision, matrices de confusion, courbes prédites, scatter prédit vs réel, barres de temps, barres de speedup, boxplots de variabilité

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
