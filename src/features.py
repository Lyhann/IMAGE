import numpy as np


def extraire_caracteristiques_patch(patch):
    """
    Transforme un patch en vecteur de caractéristiques.

    Pour chaque bande :
    - moyenne
    - écart-type

    Entrée :
        patch : tableau (bandes, h, w)

    Sortie :
        vecteur : tableau 1D
    """
    patch = patch.astype(np.float32)

    nb_bandes = patch.shape[0]
    patch_aplati = patch.reshape(nb_bandes, -1)

    moyennes = patch_aplati.mean(axis=1)
    ecarts_types = patch_aplati.std(axis=1)

    vecteur = np.concatenate([moyennes, ecarts_types])

    return vecteur