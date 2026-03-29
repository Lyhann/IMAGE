import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from config import (
    chemin_image,
    taille_patch,
    pas_deplacement,
    regions_apprentissage_batiments,
    regions_apprentissage_non_batiments,
    regions_evaluation_batiments
)

from io_utils import charger_image
from dataset import extraire_region, decouper_region_en_patches
from features import extraire_caracteristiques_patch

def preparer_donnees(liste_regions, label, image):
    """
    Extrait les caractéristiques et les labels pour une liste de régions.
    """
    X = []
    y = []
    for (x, y_coord, w, h) in liste_regions:
        region = extraire_region(image, x, y_coord, w, h)
        patches = decouper_region_en_patches(region, taille_patch, pas_deplacement)
        for patch in patches:
            vecteur = extraire_caracteristiques_patch(patch)
            X.append(vecteur)
            y.append(label)
    return X, y

def main():
    # 1. Chargement de l'image (via le XML DIMAP)
    print(f"Chargement de l'image : {chemin_image}")
    image, _ = charger_image(chemin_image)
    print(f"Image chargée. Dimensions : {image.shape}")

    # 2. Préparation des données d'apprentissage (Point n°1)
    print("\nExtraction des données d'apprentissage...")
    X_bat, y_bat = preparer_donnees(regions_apprentissage_batiments, 1, image)
    X_non_bat, y_non_bat = preparer_donnees(regions_apprentissage_non_batiments, 0, image)
    
    X_train = np.array(X_bat + X_non_bat)
    y_train = np.array(y_bat + y_non_bat)
    print(f"Taille de l'ensemble d'entraînement : {len(X_train)} patches")

    # 3. Évaluation de l'influence de K (Point n°5)
    # On teste différentes valeurs pour justifier le choix final
    valeurs_k = [1, 3, 5, 7, 11]
    
    for k in valeurs_k:
        print(f"\n--- Évaluation avec K = {k} ---")
        modele = KNeighborsClassifier(n_neighbors=k)
        modele.fit(X_train, y_train)

        # 4. Test sur les régions d'évaluation (Point n°2)
        # On applique ici la réduction de surface (Point n°4)
        print("Test sur les régions d'évaluation avec filtre NDVI...")
        
        y_true_all = []
        y_pred_all = []

        for (x, y_c, w, h) in regions_evaluation_batiments:
            region_test = extraire_region(image, x, y_c, w, h)
            patches_test = decouper_region_en_patches(region_test, taille_patch, pas_deplacement)
            
            for p in patches_test:
                # Point n°4 : Identifier les régions caractéristiques (Végétation) de manière certaine
                # Si le NDVI est élevé, on classe en 'Non-Bâtiment' (0) sans utiliser le K-NN
                r, pir = np.mean(p[0]), np.mean(p[3])
                ndvi = (pir - r) / (pir + r + 1e-8)
                
                y_true_all.append(1) # On suppose que ces régions de test sont des bâtiments
                
                if ndvi > 0.5: # Seuil de végétation (Réduction de calcul)
                    y_pred_all.append(0)
                else:
                    # Utilisation du modèle K-NN seulement si le doute subsiste
                    feat = extraire_caracteristiques_patch(p).reshape(1, -1)
                    y_pred_all.append(modele.predict(feat)[0])

        # 5. Affichage des résultats (Métriques demandées)
        print(classification_report(y_true_all, y_pred_all, target_names=['Non-Bâtiment', 'Bâtiment'], zero_division=0))

if __name__ == "__main__":
    main()