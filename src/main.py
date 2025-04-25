import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from config import *
from utils.texts import TEXTS
from utils.bbox_utils import (
    create_overlay,
    filter_contours,
    filter_contained_boxes
)
from utils.gui import BBoxGUI

class BBoxApp:
    def __init__(self):
        self.root = tk.Tk()
        self.gui = BBoxGUI(
            self.root,
            on_validate=self.validate_box,
            on_reject=self.reject_box,
            on_quit=self.quit_app,
            on_save=self.save_results
        )
        
        # Variables d'état
        self.bounding_boxes = {}
        self.bad_cases = []
        self.validations = []
        self.current_image = None
        self.current_mask = None
        self.current_boxes = None
        self.current_scan = None
        self.current_img_name = None
        self.is_running = True
        
        # Variables de cache pour les calculs
        self.last_alpha = None
        self.last_mask_state = None
        self.last_image_hash = None
        
        # Chargement des validations existantes
        if os.path.exists("validations.json"):
            with open("validations.json", 'r', encoding='utf-8') as f:
                self.validations = json.load(f)
                
        # Variables de contrôle
        self.validation_var = tk.BooleanVar()
        self.validation_var.set(False)
        
        # Configuration de la mise à jour automatique
        self.update_interval = 800 #ms
        self.root.after(self.update_interval, self.update_interface)
                
    def run(self):
        """Lance l'application"""
        self.process_next_image()
        self.root.mainloop()
        
    def process_next_image(self):
        """Traite la prochaine image"""
        if not self.is_running:
            return
            
        # Trouver la prochaine image à traiter
        for scan_folder in os.listdir(IMAGE_BASE_DIR):
            scan_path = os.path.join(IMAGE_BASE_DIR, scan_folder)
            if not os.path.isdir(scan_path):
                continue
                
            for img_name in sorted(os.listdir(scan_path)):
                if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                key_str = f"{scan_folder}/{img_name}"
                if key_str in self.bounding_boxes or key_str in self.bad_cases:
                    continue
                    
                self.current_scan = scan_folder
                self.current_img_name = img_name
                
                img_path = os.path.join(scan_path, img_name)
                mask_path = os.path.join(MASK_BASE_DIR, scan_folder, img_name)
                
                self.current_image = cv2.imread(img_path)
                self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if self.current_image is None or self.current_mask is None:
                    print(f"{TEXTS[self.gui.current_lang]['read_error']}: {scan_folder}/{img_name}")
                    self.bad_cases.append(key_str)
                    self.validations.append({"scan": scan_folder, "image": img_name, "valid": False})
                    continue
                    
                # Redimensionner le masque
                self.current_mask = cv2.resize(self.current_mask, 
                                             (self.current_image.shape[1], self.current_image.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                _, mask_bin = cv2.threshold(self.current_mask, 127, 255, cv2.THRESH_BINARY)
                
                # Détection et filtrage des contours
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filtered_contours = filter_contours(contours, MIN_AREA)
                
                if not filtered_contours:
                    print(f"{TEXTS[self.gui.current_lang]['no_object']}: {scan_folder}/{img_name}")
                    self.bad_cases.append(key_str)
                    self.validations.append({"scan": scan_folder, "image": img_name, "valid": False})
                    continue
                    
                # Extraire les bounding boxes
                boxes = []
                for contour in filtered_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    boxes.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
                    
                # Filtrer les boxes
                self.current_boxes = filter_contained_boxes(boxes)
                
                if not self.current_boxes:
                    print(f"{TEXTS[self.gui.current_lang]['no_object']}: {scan_folder}/{img_name}")
                    self.bad_cases.append(key_str)
                    self.validations.append({"scan": scan_folder, "image": img_name, "valid": False})
                    continue
                    
                # Réinitialiser le cache
                self.last_alpha = None
                self.last_mask_state = None
                self.last_image_hash = None
                
                # Mise à jour de l'interface
                self.update_interface()
                
                # Attendre la validation
                self.validation_var.set(False)
                self.root.wait_variable(self.validation_var)
                
                if not self.is_running:
                    return
                    
        # Toutes les images ont été traitées
        messagebox.showinfo("Terminé", "Toutes les images ont été traitées.")
        self.quit_app()
        
    def update_interface(self):
        """Met à jour l'interface avec l'image courante"""
        if not self.current_image is None and not self.current_mask is None:
            # Vérifier si une mise à jour est nécessaire
            current_alpha = self.gui.current_alpha if self.gui.mask_enabled else 0
            current_hash = hash(self.current_image.tobytes())
            
            if (current_alpha != self.last_alpha or 
                self.gui.mask_enabled != self.last_mask_state or
                current_hash != self.last_image_hash):
                
                # Création des visualisations
                overlay = create_overlay(self.current_image, self.current_mask, 
                                       color=(0, 0, 255), alpha=current_alpha)
                overlay = create_overlay(overlay, ~self.current_mask, 
                                       color=(0, 255, 0), alpha=current_alpha)
                
                # Image avec bounding boxes
                bbox_img = self.current_image.copy()
                for box in self.current_boxes:
                    cv2.rectangle(bbox_img,
                                 (box["x"], box["y"]),
                                 (box["x"] + box["width"], box["y"] + box["height"]),
                                 (0, 255, 0), 2)
                                 
                # Mise à jour de l'interface
                self.gui.update_image(overlay, bbox_img)
                self.gui.update_info(
                    self.current_scan,
                    self.current_img_name,
                    len(self.current_boxes),
                    image_size=(self.current_image.shape[1], self.current_image.shape[0]),
                    boxes=self.current_boxes,
                    mask=self.current_mask
                )
                
                # Mettre à jour le cache
                self.last_alpha = current_alpha
                self.last_mask_state = self.gui.mask_enabled
                self.last_image_hash = current_hash
            
        # Planifier la prochaine mise à jour
        self.root.after(self.update_interval, self.update_interface)
        
    def validate_box(self):
        """Valide la bounding box courante"""
        key_str = f"{self.current_scan}/{self.current_img_name}"
        self.bounding_boxes[key_str] = self.current_boxes
        self.validations.append({"scan": self.current_scan, 
                               "image": self.current_img_name, 
                               "valid": True})
        self.save_results()
        self.validation_var.set(True)
        
    def reject_box(self):
        """Rejette la bounding box courante"""
        key_str = f"{self.current_scan}/{self.current_img_name}"
        self.bad_cases.append(key_str)
        self.validations.append({"scan": self.current_scan, 
                               "image": self.current_img_name, 
                               "valid": False})
        self.save_results()
        self.validation_var.set(True)
        
    def save_results(self):
        """Sauvegarde les résultats"""
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(self.bounding_boxes, f, indent=2, ensure_ascii=False)
        with open("validations.json", 'w', encoding='utf-8') as f:
            json.dump(self.validations, f, indent=2, ensure_ascii=False)
            
        # Création du DataFrame
        df_rows = []
        for img_path, boxes in self.bounding_boxes.items():
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
        df.to_csv(OUTPUT_CSV, index=False)
        
    def quit_app(self):
        """Quitte l'application"""
        self.is_running = False
        self.validation_var.set(True)
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    app = BBoxApp()
    app.run() 