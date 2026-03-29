import rasterio
import matplotlib.pyplot as plt
import numpy as np

def charger_image(chemin):
    with rasterio.open(chemin) as src:
        image = src.read()
        meta = src.meta
    return image, meta

def normaliser_bande(bande):
    masque = bande > 0
    if not np.any(masque): return np.zeros_like(bande, dtype=np.float32)
    p2, p98 = np.percentile(bande[masque], (2, 98))
    bande_norm = np.clip(bande, p2, p98)
    return ((bande_norm - p2) / (p98 - p2)).astype(np.float32)

def afficher_region_rgb(region, titre="Extrait RGB"):
    r = normaliser_bande(region[0])
    g = normaliser_bande(region[1])
    b = normaliser_bande(region[2])
    plt.imshow(np.dstack((r, g, b)))
    plt.title(titre)
    plt.axis("off")
    plt.show()

def afficher_region_infrarouge(region):
    # PIR=3, R=0, V=1 -> La végétation apparaît en rouge
    pir = normaliser_bande(region[3])
    r = normaliser_bande(region[0])
    g = normaliser_bande(region[1])
    plt.imshow(np.dstack((pir, r, g)))
    plt.title("Vue Infrarouge (Rouge = Végétation)")
    plt.axis("off")
    plt.show()