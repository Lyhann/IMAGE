import numpy as np

def extraire_caracteristiques_patch(patch):
    patch = patch.astype(np.float32)
    
    # Statistiques classiques (Moyenne/STD par bande)
    moyennes = patch.mean(axis=(1, 2))
    stds = patch.std(axis=(1, 2))
    
    # Point n°3 : Ajout du NDVI moyen du patch
    # Formule : (PIR - R) / (PIR + R)
    rouge = patch[0]
    pir = patch[3]
    ndvi = (pir - rouge) / (pir + rouge + 1e-8)
    ndvi_moyen = np.mean(ndvi)
    
    return np.concatenate([moyennes, stds, [ndvi_moyen]])