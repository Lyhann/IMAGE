# src/model_knn.py

from sklearn.neighbors import KNeighborsClassifier


def entrainer_modele_knn(X_entrainement, y_entrainement, k=3):
    """
    Entraîne un modèle KNN.

    Entrées :
        X_entrainement : données d'entraînement
        y_entrainement : labels d'entraînement
        k : nombre de voisins

    Sortie :
        modele : modèle entraîné
    """
    modele = KNeighborsClassifier(n_neighbors=k)
    modele.fit(X_entrainement, y_entrainement)

    return modele