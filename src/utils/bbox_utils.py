import cv2
import numpy as np

def create_overlay(image, mask, color=(0, 255, 0), alpha=0.3):
    """
    Crée une superposition du masque sur l'image avec transparence.
    
    Args:
        image: Image originale
        mask: Masque binaire
        color: Couleur du masque (BGR)
        alpha: Transparence (0-1)
    """
    if alpha == 0:
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
    area1 = get_box_area(box1)
    area2 = get_box_area(box2)
    
    if is_box_contained(box1, box2):
        return True
    
    intersection = get_intersection_area(box1, box2)
    
    if intersection > 0 and area1 < area2:
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
        
        for j in range(n):
            if i != j and should_remove_box(box_i, boxes[j]):
                should_remove = True
                break
                
        if not should_remove:
            filtered_boxes.append(box_i)
            
    return filtered_boxes 