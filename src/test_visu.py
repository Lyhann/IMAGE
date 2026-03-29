from io_utils import charger_image, afficher_image_rgb, afficher_patch
from dataset import extraire_region
from config import chemin_image, regions_apprentissage_batiments

# 1. Charger les métadonnées et le pointeur d'image
image, meta = charger_image(chemin_image)
print(f"Dimensions totales : {image.shape}")

# 2. Extraire une petite région au lieu de TOUTE l'image
# On prend la première région de bâtiments définie dans ton config.py
x, y, w, h = regions_apprentissage_batiments[0]
petite_region = extraire_region(image, x, y, w, h)

print(f"Dimensions de l'extrait : {petite_region.shape}")

# 3. Afficher uniquement cet extrait
afficher_image_rgb(petite_region)