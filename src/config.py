# Paramètres généraux

chemin_image = r"data/raw/image.TIF"

taille_patch = 16
pas_deplacement = 16


k_voisins = 3
taille_test = 0.3
graine_aleatoire = 42

regions_apprentissage_batiments = [
    (9642, 5692, 256, 256),
    (6072, 5302, 256, 256),
]

regions_apprentissage_non_batiments = [
    (4382, 12002, 256, 256),
    (10742, 612, 256, 256),
]
regions_evaluation_batiments = [
    (8792, 6102, 256, 256),
    (8152, 13362, 256, 256),
]