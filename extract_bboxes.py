import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Ajout des textes multilingues
TEXTS = {
    'fr': {
        'scan': 'Scan',
        'image': 'Image',
        'controls': """
        Controles:
        Y: Valider | N: Rejeter | Q: Quitter
        W/S: Ajuster l'opacite
        L: Changer la langue
        M: Activer/Desactiver le masque
        H: Afficher/Masquer l'aide
        """,
        'processing': 'Traitement du scan',
        'no_object': 'Aucun objet trouve dans le masque pour',
        'read_error': 'Erreur de lecture pour',
        'finished': 'Termine!',
        'saved_boxes': 'bounding boxes sauvegardees sur',
        'processed_images': 'images traitees',
        'manual_fix': 'cas à corriger manuellement. Voir'
    },
    'en': {
        'scan': 'Scan',
        'image': 'Image',
        'controls': """
        Controls:
        Y: Validate | N: Reject | Q: Quit
        W/S: Adjust opacity
        L: Change language
        M: Toggle mask
        H: Show/Hide help
        """,
        'processing': 'Processing scan',
        'no_object': 'No object found in mask for',
        'read_error': 'Read error for',
        'finished': 'Finished!',
        'saved_boxes': 'bounding boxes saved out of',
        'processed_images': 'processed images',
        'manual_fix': 'cases to fix manually. See'
    },
    'sv': {
        'scan': 'Skanning',
        'image': 'Bild',
        'controls': """
        Kontroller:
        Y: Godkann | N: Avvisa | Q: Avsluta
        W/S: Justera opacitet
        L: Byt sprak
        M: Vaxla mask
        H: Visa/Dolj hjalp
        """,
        'processing': 'Bearbetar skanning',
        'no_object': 'Inget objekt hittat i masken for',
        'read_error': 'Lasfel for',
        'finished': 'Fardig!',
        'saved_boxes': 'markeringsrutor sparade av',
        'processed_images': 'bearbetade bilder',
        'manual_fix': 'fall att fixa manuellt. Se'
    }
}

def create_overlay(image, mask, color=(0, 255, 0), alpha=0.3):
    """
    Crée une superposition du masque sur l'image avec transparence.
    
    Args:
        image: Image originale
        mask: Masque binaire
        color: Couleur du masque (BGR)
        alpha: Transparence (0-1)
    """
    if alpha == 0:  # Si l'opacité est 0, retourner l'image sans modification
        return image.copy()
    overlay = image.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
    return overlay

def filter_contours(contours, min_area=100):
    """
    Filtre les contours selon leur aire.
    
    Args:
        contours: Liste des contours
        min_area: Aire minimale pour conserver un contour
    Returns:
        Liste des contours filtrés
    """
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def is_box_contained(box1, box2):
    """
    Vérifie si box1 est complètement contenue dans box2.
    
    Args:
        box1, box2: Dictionnaires contenant x, y, width, height
    Returns:
        True si box1 est complètement contenue dans box2, False sinon
    """
    x1_1, y1_1 = box1['x'], box1['y']
    x2_1, y2_1 = x1_1 + box1['width'], y1_1 + box1['height']
    
    x1_2, y1_2 = box2['x'], box2['y']
    x2_2, y2_2 = x1_2 + box2['width'], y1_2 + box2['height']
    
    return (x1_1 >= x1_2 and y1_1 >= y1_2 and 
            x2_1 <= x2_2 and y2_1 <= y2_2)

def get_box_area(box):
    """
    Calcule l'aire d'une bounding box.
    """
    return box['width'] * box['height']

def get_intersection_area(box1, box2):
    """
    Calcule l'aire d'intersection entre deux bounding boxes.
    """
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
    
    if x2 <= x1 or y2 <= y1:
        return 0
    
    return (x2 - x1) * (y2 - y1)

def should_remove_box(box1, box2):
    """
    Détermine si box1 doit être supprimée par rapport à box2.
    
    Critères :
    1. Si box1 est complètement contenue dans box2
    2. Si box1 intersecte box2 et que l'aire de box1 est < 10% de l'aire de box2
    """
    # Calcul des aires
    area1 = get_box_area(box1)
    area2 = get_box_area(box2)
    
    # Si box1 est complètement contenue dans box2
    if is_box_contained(box1, box2):
        return True
    
    # Calcul de l'intersection
    intersection = get_intersection_area(box1, box2)
    
    # S'il y a intersection et que box1 est la plus petite
    if intersection > 0 and area1 < area2:
        # Si l'aire de box1 est inférieure à 10% de l'aire de box2
        if area1 < 0.1 * area2:
            return True
    
    return False

def filter_contained_boxes(boxes):
    """
    Filtre les bounding boxes selon les critères :
    - Complètement contenues dans une autre
    - Partiellement contenues et aire < 10% de la plus grande
    """
    if not boxes:
        return boxes
        
    filtered_boxes = []
    n = len(boxes)
    
    for i in range(n):
        box_i = boxes[i]
        should_remove = False
        
        # Vérifie les critères de suppression par rapport à toutes les autres boxes
        for j in range(n):
            if i != j and should_remove_box(box_i, boxes[j]):
                should_remove = True
                break
                
        if not should_remove:
            filtered_boxes.append(box_i)
            
    return filtered_boxes

def extract_bboxes(image_base_dir, mask_base_dir, output_json="bounding_boxes.json", output_csv="bounding_boxes.csv", bad_cases_file="to_fix.txt", min_area=100):
    """
    Extrait les bounding boxes des masques et permet leur validation manuelle.
    """
    bounding_boxes = {}
    bad_cases = []
    validations = []
    
    # Variables pour les fonctionnalités
    current_lang = 'fr'
    alpha = 0.3
    last_alpha = alpha
    show_help = True
    mask_enabled = True
    
    # Chargement des validations existantes
    if os.path.exists("validations.json"):
        with open("validations.json", 'r', encoding='utf-8') as f:
            validations = json.load(f)

    # Vérification des dossiers
    if not os.path.exists(image_base_dir):
        raise FileNotFoundError(f"Le dossier {image_base_dir} n'existe pas")
    if not os.path.exists(mask_base_dir):
        raise FileNotFoundError(f"Le dossier {mask_base_dir} n'existe pas")

    cv2.namedWindow('Validation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Validation', 1600, 800)

    total_images = 0
    for scan_folder in os.listdir(image_base_dir):
        scan_path = os.path.join(image_base_dir, scan_folder)
        if not os.path.isdir(scan_path):
            continue

        print(f"\n{TEXTS[current_lang]['processing']}: {scan_folder}")
        
        for img_name in sorted(os.listdir(scan_path)):
            if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                continue

            total_images += 1
            img_path = os.path.join(scan_path, img_name)
            mask_path = os.path.join(mask_base_dir, scan_folder, img_name)

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"{TEXTS[current_lang]['read_error']}: {scan_folder}/{img_name}")
                bad_cases.append(f"{scan_folder}/{img_name}")
                validations.append({"scan": scan_folder, "image": img_name, "valid": False})
                continue

            # Redimensionner le masque
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Détection et filtrage des contours
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = filter_contours(contours, min_area)

            if not filtered_contours:
                print(f"{TEXTS[current_lang]['no_object']}: {scan_folder}/{img_name}")
                bad_cases.append(f"{scan_folder}/{img_name}")
                validations.append({"scan": scan_folder, "image": img_name, "valid": False})
                continue

            # Extraire toutes les bounding boxes
            boxes = []
            for contour in filtered_contours:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
            
            # Filtrer les boxes contenues dans d'autres
            boxes = filter_contained_boxes(boxes)

            if not boxes:
                print(f"{TEXTS[current_lang]['no_object']}: {scan_folder}/{img_name}")
                bad_cases.append(f"{scan_folder}/{img_name}")
                validations.append({"scan": scan_folder, "image": img_name, "valid": False})
                continue

            while True:
                if mask_enabled:
                    current_alpha = alpha
                else:
                    current_alpha = 0

                # Création des visualisations
                overlay = create_overlay(image, mask_bin, color=(0, 0, 255), alpha=current_alpha)
                overlay = create_overlay(overlay, ~mask_bin, color=(0, 255, 0), alpha=current_alpha)
                
                # Image avec toutes les bounding boxes
                bbox_img = image.copy()
                for box in boxes:
                    cv2.rectangle(bbox_img, 
                                (box["x"], box["y"]), 
                                (box["x"] + box["width"], box["y"] + box["height"]), 
                                (0, 255, 0), 2)

                combined = np.hstack((overlay, bbox_img))

                # Ajout du texte
                cv2.putText(combined, f"{TEXTS[current_lang]['scan']}: {scan_folder}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, f"{TEXTS[current_lang]['image']}: {img_name}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, f"Boxes: {len(boxes)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if show_help:
                    y_pos = 120
                    for line in TEXTS[current_lang]['controls'].split('\n'):
                        cv2.putText(combined, line, (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        y_pos += 30

                cv2.imshow('Validation', combined)
                key = cv2.waitKey(1) & 0xFF

                # Gestion des touches
                if key == ord('y') or key == ord('Y'):
                    key_str = f"{scan_folder}/{img_name}"
                    bounding_boxes[key_str] = boxes
                    validations.append({"scan": scan_folder, "image": img_name, "valid": True})
                    
                    # Sauvegarde immédiate
                    with open(output_json, 'w', encoding='utf-8') as f:
                        json.dump(bounding_boxes, f, indent=2, ensure_ascii=False)
                    with open("validations.json", 'w', encoding='utf-8') as f:
                        json.dump(validations, f, indent=2, ensure_ascii=False)
                    break

                elif key == ord('n') or key == ord('N'):
                    bad_cases.append(f"{scan_folder}/{img_name}")
                    validations.append({"scan": scan_folder, "image": img_name, "valid": False})
                    
                    with open(bad_cases_file, 'w', encoding='utf-8') as f:
                        for case in bad_cases:
                            f.write(case + "\n")
                    with open("validations.json", 'w', encoding='utf-8') as f:
                        json.dump(validations, f, indent=2, ensure_ascii=False)
                    break

                elif key == ord('q') or key == ord('Q'):
                    cv2.destroyAllWindows()
                    return

                # Nouvelles touches - Modification ici
                elif key == ord('w') or key == ord('W'):  # W pour augmenter l'opacité
                    alpha = min(1.0, alpha + 0.1)
                elif key == ord('s') or key == ord('S'):  # S pour diminuer l'opacité
                    alpha = max(0.0, alpha - 0.1)
                elif key == ord('l') or key == ord('L'):
                    # Rotation des langues
                    langs = list(TEXTS.keys())
                    current_lang = langs[(langs.index(current_lang) + 1) % len(langs)]
                elif key == ord('m') or key == ord('M'):
                    if mask_enabled:
                        mask_enabled = False
                        last_alpha = alpha
                    else:
                        mask_enabled = True
                        alpha = last_alpha
                elif key == ord('h') or key == ord('H'):
                    show_help = not show_help

    cv2.destroyAllWindows()

    # Création du DataFrame avec toutes les boxes
    df_rows = []
    for img_path, boxes in bounding_boxes.items():
        scan_folder, img_name = img_path.split('/')
        for box in boxes:
            df_rows.append({
                'scan_folder': scan_folder,
                'image_name': img_name,
                'x': box['x'],
                'y': box['y'],
                'width': box['width'],
                'height': box['height']
            })
    
    df = pd.DataFrame(df_rows)
    df.to_csv(output_csv, index=False)

    print(f"\n{TEXTS[current_lang]['finished']} {len(bounding_boxes)} {TEXTS[current_lang]['saved_boxes']} {total_images} {TEXTS[current_lang]['processed_images']}.")
    print(f"{len(bad_cases)} {TEXTS[current_lang]['manual_fix']} '{bad_cases_file}'.")

if __name__ == "__main__":
    IMAGE_BASE_DIR = "Miccai 2022 BUV Dataset/rawframes/benign"
    MASK_BASE_DIR = "masks/benign"
    
    extract_bboxes(IMAGE_BASE_DIR, MASK_BASE_DIR, min_area=100) 