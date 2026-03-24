# src/dataset.py

import numpy as np


def extraire_patch(image, x, y, taille_patch):
    """
    Extrait un patch carré dans l'image.

    Entrées :
        image : tableau (bandes, hauteur, largeur)
        x : position horizontale
        y : position verticale
        taille_patch : taille du patch

    Sortie :
        patch : tableau (bandes, taille_patch, taille_patch)
    """
    return image[:, y:y + taille_patch, x:x + taille_patch]


def patch_vide(patch, seuil_noir=0):
    """
    Vérifie si un patch est complètement noir.

    Entrées :
        patch : tableau numpy
        seuil_noir : seuil utilisé pour définir le noir

    Sortie :
        True ou False
    """
    return np.max(patch) <= seuil_noir


def proportion_noire(patch, seuil_noir=0):
    """
    Calcule la proportion de pixels noirs dans un patch.

    Un pixel est considéré noir si toutes ses bandes sont <= seuil_noir.

    Entrées :
        patch : tableau numpy
        seuil_noir : seuil utilisé pour définir le noir

    Sortie :
        proportion de pixels noirs entre 0 et 1
    """
    pixels_noirs = np.all(patch <= seuil_noir, axis=0)
    return np.mean(pixels_noirs)


def patch_utilisable(patch, seuil_noir=0, ratio_noir_max=0.20):
    """
    Vérifie si un patch est exploitable.

    Entrées :
        patch : tableau numpy
        seuil_noir : seuil de noir
        ratio_noir_max : proportion maximale de noir autorisée

    Sortie :
        True ou False
    """
    if patch_vide(patch, seuil_noir):
        return False

    ratio_noir = proportion_noire(patch, seuil_noir)

    if ratio_noir > ratio_noir_max:
        return False

    return True


def extraire_tous_les_patches(
    image,
    taille_patch=16,
    pas_deplacement=16,
    filtrer_vides=True,
    seuil_noir=0,
    ratio_noir_max=0.20
):
    """
    Découpe l'image en patches.

    Entrées :
        image : tableau (bandes, hauteur, largeur)
        taille_patch : taille d'un patch
        pas_deplacement : pas de déplacement de la fenêtre
        filtrer_vides : enlève ou non les patches vides
        seuil_noir : seuil de noir
        ratio_noir_max : proportion maximale de noir autorisée

    Sorties :
        liste_patches : liste des patches
        liste_positions : liste des positions (x, y)
    """
    liste_patches = []
    liste_positions = []

    _, hauteur, largeur = image.shape

    for y in range(0, hauteur - taille_patch + 1, pas_deplacement):
        for x in range(0, largeur - taille_patch + 1, pas_deplacement):
            patch = extraire_patch(image, x, y, taille_patch)

            if filtrer_vides:
                if not patch_utilisable(patch, seuil_noir, ratio_noir_max):
                    continue

            liste_patches.append(patch)
            liste_positions.append((x, y))

    return liste_patches, liste_positions