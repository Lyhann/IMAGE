import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

from config import (
    chemin_image,
    taille_patch,
    pas_deplacement,
    k_voisins,
    regions_apprentissage_batiments,
    regions_apprentissage_non_batiments,
    regions_evaluation_batiments
)

from io_utils import charger_image, afficher_image_rgb, afficher_patch
from dataset import construire_patches_depuis_regions
from features import extraire_caracteristiques_patch


def construire_X_y(patches_batiments, patches_non_batiments):
    """
    Construit les données d'apprentissage.

    Entrées :
        patches_batiments : liste de patches bâtiment
        patches_non_batiments : liste de patches non bâtiment

    Sorties :
        X : matrice des caractéristiques
        y : labels
    """
    X = []
    y = []

    for patch in patches_batiments:
        vecteur = extraire_caracteristiques_patch(patch)
        X.append(vecteur)
        y.append(1)

    for patch in patches_non_batiments:
        vecteur = extraire_caracteristiques_patch(patch)
        X.append(vecteur)
        y.append(0)

    return np.array(X), np.array(y)


def construire_X(patches):
    """
    Transforme une liste de patches en matrice de caractéristiques.
    """
    X = []

    for patch in patches:
        vecteur = extraire_caracteristiques_patch(patch)
        X.append(vecteur)

    return np.array(X)


def main():
    # 1. Chargement de l'image
    image, meta = charger_image(chemin_image)

    print("Shape image :", image.shape)
    ##print("Meta :", meta)

    afficher_image_rgb(image)

    # 2. Découpage des régions d'apprentissage
    patches_apprentissage_batiments = construire_patches_depuis_regions(
        image,
        regions_apprentissage_batiments,
        taille_patch,
        pas_deplacement
    )

    patches_apprentissage_non_batiments = construire_patches_depuis_regions(
        image,
        regions_apprentissage_non_batiments,
        taille_patch,
        pas_deplacement
    )

    # 3. Affichage de quelques patches d'exemple
    if len(patches_apprentissage_batiments) > 0:
        afficher_patch(patches_apprentissage_batiments[0], "Exemple patch apprentissage bâtiment")

    if len(patches_apprentissage_non_batiments) > 0:
        afficher_patch(patches_apprentissage_non_batiments[0], "Exemple patch apprentissage non bâtiment")

    print("Nombre patches apprentissage bâtiment :", len(patches_apprentissage_batiments))
    print("Nombre patches apprentissage non bâtiment :", len(patches_apprentissage_non_batiments))

    # 4. Construction des données d'apprentissage
    X_train, y_train = construire_X_y(
        patches_apprentissage_batiments,
        patches_apprentissage_non_batiments
    )

    print("Shape X_train :", X_train.shape)
    print("Shape y_train :", y_train.shape)

    # 5. Entraînement du modèle
    modele = KNeighborsClassifier(n_neighbors=k_voisins)
    modele.fit(X_train, y_train)

    # 6. Découpage des régions d'évaluation contenant des bâtiments
    patches_evaluation_batiments = construire_patches_depuis_regions(
        image,
        regions_evaluation_batiments,
        taille_patch,
        pas_deplacement
    )

    print("Nombre patches évaluation bâtiment :", len(patches_evaluation_batiments))

    # 7. Transformation en données de test
    X_test = construire_X(patches_evaluation_batiments)

    # 8. Comme ces régions contiennent des bâtiments,
    # on considère que les labels attendus sont 1
    y_test = np.ones(len(X_test), dtype=int)

    # 9. Prédictions
    y_prediction = modele.predict(X_test)

    print("y_test :", y_test)
    print("y_prediction :", y_prediction)

    # 10. Évaluation
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_prediction))

    print("\nRapport de classification :")
    print(classification_report(y_test, y_prediction, digits=4, zero_division=0))


if __name__ == "__main__":
    main()