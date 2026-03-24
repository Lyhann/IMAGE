import rasterio
import matplotlib.pyplot as plt
import numpy as np


def charger_image(chemin):
    """
    Charge l'image satellite.

    Entrée :
        chemin : chemin du fichier image

    Sortie :
        image : tableau numpy (bandes, hauteur, largeur)
        meta : métadonnées
    """
    with rasterio.open(chemin) as src:
        image = src.read()
        meta = src.meta

    return image, meta


def normaliser_bande(bande):
    """
    Normalise une bande entre 0 et 1 pour l'affichage.
    """
    p2, p98 = np.percentile(bande, (2, 98))

    if p98 == p2:
        return np.zeros_like(bande, dtype=np.float32)

    bande = np.clip(bande, p2, p98)
    bande = (bande - p2) / (p98 - p2)

    return bande.astype(np.float32)


def afficher_image_rgb(image):
    """
    Affiche l'image avec les 3 premières bandes.
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


def afficher_patch(patch, titre="Patch"):
    """
    Affiche un patch.
    """
    rouge = patch[0].astype(np.float32)
    vert = patch[1].astype(np.float32)
    bleu = patch[2].astype(np.float32)

    image_rgb = np.dstack((rouge, vert, bleu))

    if image_rgb.max() > 0:
        image_rgb = image_rgb / image_rgb.max()

    plt.figure(figsize=(4, 4))
    plt.imshow(image_rgb)
    plt.title(titre)
    plt.axis("off")
    plt.show()