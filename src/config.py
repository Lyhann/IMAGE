# src/config.py

chemin_image = r"data/raw/image.TIF"

taille_patch = 16
pas_deplacement = 16

filtrer_patches_vides = True
seuil_noir = 0
ratio_noir_max = 0.20

k_voisins = 3
taille_test = 0.3
graine_aleatoire = 42

# A modifier après observation visuelle des patches
indices_batiments = [3000, 12000, 20000, 25000, 30000]
indices_non_batiments = [100, 500, 8000, 15000, 22000]