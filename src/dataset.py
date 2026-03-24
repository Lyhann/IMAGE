import numpy as np


def extraire_region(image, x, y, largeur, hauteur):
    """
    Extrait une région rectangulaire de l'image.

    Entrées :
        image : tableau (bandes, hauteur, largeur)
        x, y : coin haut gauche
        largeur, hauteur : dimensions de la région

    Sortie :
        region : tableau (bandes, hauteur, largeur)
    """
    return image[:, y:y + hauteur, x:x + largeur]


def extraire_patch(image, x, y, taille_patch):
    """
    Extrait un patch carré.
    """
    return image[:, y:y + taille_patch, x:x + taille_patch]


def decouper_region_en_patches(region, taille_patch=16, pas_deplacement=16):
    """
    Découpe une région en patches.

    Entrées :
        region : tableau (bandes, hauteur, largeur)
        taille_patch : taille du patch
        pas_deplacement : pas de déplacement

    Sorties :
        liste_patches : liste des patches
    """
    liste_patches = []

    _, hauteur, largeur = region.shape

    for y in range(0, hauteur - taille_patch + 1, pas_deplacement):
        for x in range(0, largeur - taille_patch + 1, pas_deplacement):
            patch = extraire_patch(region, x, y, taille_patch)
            liste_patches.append(patch)

    return liste_patches


def construire_patches_depuis_regions(image, liste_regions, taille_patch=16, pas_deplacement=16):
    """
    Extrait toutes les régions puis les découpe en patches.

    Entrées :
        image : image complète
        liste_regions : liste de rectangles (x, y, largeur, hauteur)
        taille_patch : taille du patch
        pas_deplacement : pas de déplacement

    Sortie :
        tous_les_patches : liste de patches
    """
    tous_les_patches = []

    for (x, y, largeur, hauteur) in liste_regions:
        region = extraire_region(image, x, y, largeur, hauteur)
        patches_region = decouper_region_en_patches(region, taille_patch, pas_deplacement)
        tous_les_patches.extend(patches_region)

    return tous_les_patches