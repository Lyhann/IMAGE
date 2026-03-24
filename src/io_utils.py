# src/io_utils.py

import rasterio
import matplotlib.pyplot as plt
import numpy as np


def charger_image(chemin):
    """
    Charge une image satellite.

    Entrée :
        chemin : chemin vers l'image

    Sortie :
        image : tableau numpy de forme (bandes, hauteur, largeur)
        meta : métadonnées de l'image
    """
    with rasterio.open(chemin) as src:
        image = src.read()
        meta = src.meta

    return image, meta


def normaliser_bande(bande):
    """
    Normalise une bande entre 0 et 1 pour un meilleur affichage.

    Entrée :
        bande : image 2D

    Sortie :
        bande_normalisee : image 2D normalisée
    """
    p2, p98 = np.percentile(bande, (2, 98))

    if p98 == p2:
        return np.zeros_like(bande, dtype=np.float32)

    bande = np.clip(bande, p2, p98)
    bande_normalisee = (bande - p2) / (p98 - p2)

    return bande_normalisee.astype(np.float32)


def afficher_image_rgb(image):
    """
    Affiche l'image en utilisant les 3 premières bandes.

    Entrée :
        image : tableau numpy de forme (bandes, hauteur, largeur)
    """
    rouge = normaliser_bande(image[0])
    vert = normaliser_bande(image[1])
    bleu = normaliser_bande(image[2])

    image_rgb = np.dstack((rouge, vert, bleu))

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.title("Image RGB")
    plt.axis("off")
    plt.show()


def afficher_patch(patch, position=None, indice=None):
    """
    Affiche un patch en RGB.

    Entrées :
        patch : tableau numpy de forme (bandes, h, w)
        position : tuple (x, y)
        indice : indice du patch dans la liste
    """
    rouge = patch[0].astype(np.float32)
    vert = patch[1].astype(np.float32)
    bleu = patch[2].astype(np.float32)

    image_rgb = np.dstack((rouge, vert, bleu))

    if image_rgb.max() > 0:
        image_rgb = image_rgb / image_rgb.max()

    titre = "Patch"

    if position is not None:
        titre += f" à la position {position}"

    if indice is not None:
        titre += f" | indice {indice}"

    plt.figure(figsize=(4, 4))
    plt.imshow(image_rgb)
    plt.title(titre)
    plt.axis("off")
    plt.show()