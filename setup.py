# setup.py
# --------
# Script de compilation du fichier Cython rf_cython.pyx → rf_cython.so
#
# Usage (ligne de commande) :
#   python setup.py build_ext --inplace
#
# Cela génère un fichier rf_cython.so (Linux/Mac) ou rf_cython.pyd (Windows)
# que Python peut importer directement avec : import rf_cython

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Déclaration de l'extension Cython
# - "rf_cython" = nom du module Python après compilation
# - "rf_cython.pyx" = fichier source Cython
# - include_dirs : chemin vers les headers NumPy (pour cimport numpy)
extensions = [
    Extension(
        name="rf_cython",
        sources=["rf_cython.pyx"],
        include_dirs=[np.get_include()],  # nécessaire pour "cimport numpy"
    )
]

setup(
    name="rf_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",   # Python 3
            "boundscheck": False,     # désactive la vérification des bornes → plus rapide
            "wraparound": False,      # désactive les indices négatifs → plus rapide
        }
    ),
)
