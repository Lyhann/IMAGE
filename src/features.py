import numpy as np


def extraire_caracteristiques_patch(patch):
    """
    Extrait des caractéristiques simples d'un patch.

    Pour chaque bande :
    - moyenne
    - écart-type

    Entrée :
        patch : tableau numpy de forme (bandes, h, w)

    Sortie :
        caracteristiques : vecteur 1D
    """
    patch = patch.astype(np.float32)

    nb_bandes = patch.shape[0]

    patch_aplati = patch.reshape(nb_bandes, -1)

    moyennes = patch_aplati.mean(axis=1)
    ecarts_types = patch_aplati.std(axis=1)

    caracteristiques = np.concatenate([moyennes, ecarts_types])

    return caracteristiques