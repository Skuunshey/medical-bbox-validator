import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

class BBoxGUI:
    def __init__(self, root, on_validate, on_reject, on_quit, on_save):
        self.root = root
        self.root.title("Validation des Bounding Boxes")
        
        # Callbacks
        self.on_validate = on_validate
        self.on_reject = on_reject
        self.on_quit = on_quit
        self.on_save = on_save
        
        # Variables d'état
        self.current_alpha = 0.3
        self.mask_enabled = True
        self.show_help = True
        self.current_lang = 'fr'
        
        # Configuration de la fenêtre
        self.setup_window()
        self.create_menu()
        self.create_widgets()
        
        # Bindings clavier
        self.setup_keyboard_bindings()
        
    def setup_window(self):
        """Configure la fenêtre principale"""
        self.root.geometry("1600x800")
        self.root.minsize(800, 600)
        
    def create_menu(self):
        """Crée la barre de menus"""
        menubar = tk.Menu(self.root)
        
        # Menu File
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Quitter", command=self.handle_quit, accelerator="Q")
        file_menu.add_command(label="Sauvegarder", command=self.handle_save, accelerator="Ctrl+S")
        menubar.add_cascade(label="Fichier", menu=file_menu)
        
        # Menu Edit
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Valider", command=self.handle_validate, accelerator="Y")
        edit_menu.add_command(label="Rejeter", command=self.handle_reject, accelerator="N")
        edit_menu.add_separator()
        edit_menu.add_command(label="Augmenter transparence", command=lambda: self.handle_alpha(0.1), accelerator="W")
        edit_menu.add_command(label="Diminuer transparence", command=lambda: self.handle_alpha(-0.1), accelerator="S")
        menubar.add_cascade(label="Édition", menu=edit_menu)
        
        # Menu View
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Afficher/Masquer masque", command=self.handle_toggle_mask, accelerator="M")
        view_menu.add_command(label="Changer langue", command=self.handle_switch_language, accelerator="L")
        view_menu.add_command(label="Afficher/Masquer aide", command=self.handle_toggle_help, accelerator="H")
        menubar.add_cascade(label="Affichage", menu=view_menu)
        
        # Menu Help
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Raccourcis clavier", command=self.show_shortcuts)
        help_menu.add_command(label="À propos", command=self.show_about)
        menubar.add_cascade(label="Aide", menu=help_menu)
        
        self.root.config(menu=menubar)
        
    def create_widgets(self):
        """Crée les widgets de l'interface"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame pour les images
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Labels pour les images
        self.overlay_label = ttk.Label(self.image_frame)
        self.overlay_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.bbox_label = ttk.Label(self.image_frame)
        self.bbox_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame pour les informations
        self.info_frame = ttk.Frame(main_frame)
        self.info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Informations de base
        self.scan_label = ttk.Label(self.info_frame, text="")
        self.scan_label.pack(side=tk.LEFT, padx=5)
        
        self.image_label = ttk.Label(self.info_frame, text="")
        self.image_label.pack(side=tk.LEFT, padx=5)
        
        self.boxes_label = ttk.Label(self.info_frame, text="")
        self.boxes_label.pack(side=tk.LEFT, padx=5)
        
        # Informations détaillées
        self.details_frame = ttk.Frame(main_frame)
        self.details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Colonne 1 : Informations sur l'image
        self.image_details = ttk.Label(self.details_frame, text="")
        self.image_details.pack(side=tk.LEFT, padx=5)
        
        # Colonne 2 : Informations sur les bounding boxes
        self.bbox_details = ttk.Label(self.details_frame, text="")
        self.bbox_details.pack(side=tk.LEFT, padx=5)
        
        # Colonne 3 : Informations sur le masque
        self.mask_details = ttk.Label(self.details_frame, text="")
        self.mask_details.pack(side=tk.LEFT, padx=5)
        
    def setup_keyboard_bindings(self):
        """Configure les raccourcis clavier"""
        self.root.bind('q', lambda e: self.handle_quit())
        self.root.bind('Q', lambda e: self.handle_quit())
        self.root.bind('<Control-s>', lambda e: self.handle_save())
        self.root.bind('<Control-S>', lambda e: self.handle_save())
        self.root.bind('y', lambda e: self.handle_validate())
        self.root.bind('Y', lambda e: self.handle_validate())
        self.root.bind('n', lambda e: self.handle_reject())
        self.root.bind('N', lambda e: self.handle_reject())
        self.root.bind('w', lambda e: self.handle_alpha(0.1))
        self.root.bind('W', lambda e: self.handle_alpha(0.1))
        self.root.bind('s', lambda e: self.handle_alpha(-0.1))
        self.root.bind('S', lambda e: self.handle_alpha(-0.1))
        self.root.bind('m', lambda e: self.handle_toggle_mask())
        self.root.bind('M', lambda e: self.handle_toggle_mask())
        self.root.bind('l', lambda e: self.handle_switch_language())
        self.root.bind('L', lambda e: self.handle_switch_language())
        self.root.bind('h', lambda e: self.handle_toggle_help())
        self.root.bind('H', lambda e: self.handle_toggle_help())
        
    def update_image(self, overlay_img, bbox_img):
        """Met à jour l'affichage des images"""
        # Conversion OpenCV vers PIL
        overlay_pil = Image.fromarray(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
        bbox_pil = Image.fromarray(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
        
        # Redimensionnement
        overlay_pil = overlay_pil.resize((800, 600), Image.Resampling.LANCZOS)
        bbox_pil = bbox_pil.resize((800, 600), Image.Resampling.LANCZOS)
        
        # Conversion vers PhotoImage
        self.overlay_photo = ImageTk.PhotoImage(overlay_pil)
        self.bbox_photo = ImageTk.PhotoImage(bbox_pil)
        
        # Mise à jour des labels
        self.overlay_label.config(image=self.overlay_photo)
        self.bbox_label.config(image=self.bbox_photo)
        
    def update_info(self, scan_name, image_name, boxes_count, image_size=None, boxes=None, mask=None):
        """Met à jour les informations affichées"""
        # Informations de base
        self.scan_label.config(text=f"Scan: {scan_name}")
        self.image_label.config(text=f"Image: {image_name}")
        self.boxes_label.config(text=f"Boxes: {boxes_count}")
        
        # Informations détaillées
        if image_size is not None:
            # Informations sur l'image
            image_area = image_size[0] * image_size[1]
            image_info = f"Taille image: {image_size[0]}x{image_size[1]}\n"
            image_info += f"Superficie: {image_area} pixels"
            self.image_details.config(text=image_info)
            
            # Informations sur les bounding boxes
            if boxes is not None:
                bbox_info = "Superficie des boxes:\n"
                total_bbox_area = 0
                for i, box in enumerate(boxes):
                    box_area = box['width'] * box['height']
                    total_bbox_area += box_area
                    bbox_info += f"Box {i+1}: {box_area} pixels\n"
                bbox_info += f"Total: {total_bbox_area} pixels\n"
                bbox_info += f"Couvrance: {total_bbox_area/image_area*100:.2f}%"
                self.bbox_details.config(text=bbox_info)
            
            # Informations sur le masque
            if mask is not None:
                mask_area = np.sum(mask > 0)
                mask_info = f"Transparence: {self.current_alpha:.2f}\n"
                mask_info += f"Superficie masque: {mask_area} pixels\n"
                mask_info += f"Couvrance: {mask_area/image_area*100:.2f}%"
                self.mask_details.config(text=mask_info)
        
    def handle_alpha(self, delta):
        """Gère le changement de transparence"""
        self.current_alpha = max(0.0, min(1.0, self.current_alpha + delta))
        
    def handle_toggle_mask(self):
        """Gère l'activation/désactivation du masque"""
        self.mask_enabled = not self.mask_enabled
        
    def handle_switch_language(self):
        """Gère le changement de langue"""
        # Implémentation à ajouter
        pass
        
    def handle_toggle_help(self):
        """Gère l'affichage/masquage de l'aide"""
        self.show_help = not self.show_help
        
    def handle_validate(self):
        """Gère la validation"""
        if self.on_validate:
            self.on_validate()
            
    def handle_reject(self):
        """Gère le rejet"""
        if self.on_reject:
            self.on_reject()
            
    def handle_quit(self):
        """Gère la fermeture"""
        if self.on_quit:
            self.on_quit()
            
    def handle_save(self):
        """Gère la sauvegarde"""
        if self.on_save:
            self.on_save()
        
    def show_shortcuts(self):
        """Affiche les raccourcis clavier"""
        shortcuts = """
Raccourcis clavier:
Y: Valider
N: Rejeter
Q: Quitter
W: Augmenter transparence
S: Diminuer transparence
L: Changer langue
M: Activer/Désactiver masque
H: Afficher/Masquer aide
Ctrl+S: Sauvegarder
"""
        messagebox.showinfo("Raccourcis clavier", shortcuts)
        
    def show_about(self):
        """Affiche la boîte À propos"""
        messagebox.showinfo("À propos", "Application d'extraction de bounding boxes\nVersion 1.0") 