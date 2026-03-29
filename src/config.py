# config.py

# IMPORTANT : Mets le chemin complet sans les "..."
chemin_image = r"D:\Cours Univ\Image\Pleiades\DS_PHR1A_201910281410119_FR1_PX_W053N04_0922_01426-F-O\PHR_PRO_FOP4e9c938d39311_20250214144232937_1_1\IMG_PHR1A_PMS_001\DIM_PHR1A_PMS_201910281410119_ORT_PHR_PRO_FOP4e9c938d39311_20250214144232937_1_1.XML"

taille_patch = 16
pas_deplacement = 16

# Régions d'apprentissage (Point 1)
regions_apprentissage_batiments = [
    (9642, 5692, 256, 256),
    (6072, 5302, 256, 256),
]

regions_apprentissage_non_batiments = [
    (4382, 12002, 256, 256),
    (10742, 612, 256, 256),
]

# Régions d'évaluation (Point 2)
regions_evaluation_batiments = [
    (8792, 6102, 512, 512), 
]