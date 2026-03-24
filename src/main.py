# src/main.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from config import (
    chemin_image,
    taille_patch,
    pas_deplacement,
    filtrer_patches_vides,
    seuil_noir,
    ratio_noir_max,
    k_voisins,
    taille_test,
    graine_aleatoire,
    indices_batiments,
    indices_non_batiments
)

from io_utils import charger_image, afficher_image_rgb, afficher_patch
from dataset import extraire_tous_les_patches
from features import extraire_caracteristiques_patch
from model_knn import entrainer_modele_knn


def construire_donnees_supervisees(liste_patches, indices_batiments, indices_non_batiments):
    """
    Construit X et y à partir des indices choisis manuellement.

    Entrées :
        liste_patches : liste des patches
        indices_batiments : indices des patches bâtiment
        indices_non_batiments : indices des patches non bâtiment

    Sorties :
        X : matrice des caractéristiques
        y : vecteur des labels
    """
    X = []
    y = []

    for indice in indices_batiments:
        caracteristiques = extraire_caracteristiques_patch(liste_patches[indice])
        X.append(caracteristiques)
        y.append(1)

    for indice in indices_non_batiments:
        caracteristiques = extraire_caracteristiques_patch(liste_patches[indice])
        X.append(caracteristiques)
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    return X, y


def main():
    # 1. Charger l'image
    image, meta = charger_image(chemin_image)

    print("Shape image :", image.shape)
    print("Meta :", meta)

    # 2. Afficher l'image complète
    afficher_image_rgb(image)

    # 3. Découper l'image en patches
    liste_patches, liste_positions = extraire_tous_les_patches(
        image=image,
        taille_patch=taille_patch,
        pas_deplacement=pas_deplacement,
        filtrer_vides=filtrer_patches_vides,
        seuil_noir=seuil_noir,
        ratio_noir_max=ratio_noir_max
    )

    print("Nombre de patches utiles :", len(liste_patches))
    print("Shape premier patch :", liste_patches[0].shape)
    print("Position premier patch :", liste_positions[0])

    # 4. Afficher quelques patches pour vérification
    indices_a_tester = [100, 500, 1000, 3000, 8000]

    for indice in indices_a_tester:
        if indice < len(liste_patches):
            afficher_patch(liste_patches[indice], liste_positions[indice], indice)

    # 5. Tester l'extraction de caractéristiques sur un patch
    patch_test = liste_patches[100]
    vecteur_test = extraire_caracteristiques_patch(patch_test)

    print("Vecteur de caractéristiques :", vecteur_test)
    print("Taille du vecteur :", vecteur_test.shape)

    # 6. Construire le dataset supervisé
    X, y = construire_donnees_supervisees(
        liste_patches,
        indices_batiments,
        indices_non_batiments
    )

    print("Shape X :", X.shape)
    print("Shape y :", y.shape)
    print("Labels y :", y)

    # 7. Séparer apprentissage et test
    X_entrainement, X_test, y_entrainement, y_test = train_test_split(
        X,
        y,
        test_size=taille_test,
        random_state=graine_aleatoire,
        stratify=y
    )

    # 8. Entraîner le modèle
    modele = entrainer_modele_knn(X_entrainement, y_entrainement, k=k_voisins)

    # 9. Faire les prédictions
    y_prediction = modele.predict(X_test)

    print("y_test :", y_test)
    print("y_prediction :", y_prediction)

    # 10. Évaluer le modèle
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_prediction))

    print("\nRapport de classification :")
    print(classification_report(y_test, y_prediction, digits=4))


if __name__ == "__main__":
    main()