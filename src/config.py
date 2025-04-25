import os
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Chemins des dossiers
IMAGE_BASE_DIR = os.getenv('IMAGE_BASE_DIR', 'Miccai 2022 BUV Dataset/rawframes/benign')
MASK_BASE_DIR = os.getenv('MASK_BASE_DIR', 'masks/benign')

# Paramètres de traitement
MIN_AREA = int(os.getenv('MIN_AREA', 100))
OUTPUT_JSON = os.getenv('OUTPUT_JSON', 'bounding_boxes.json')
OUTPUT_CSV = os.getenv('OUTPUT_CSV', 'bounding_boxes.csv')
BAD_CASES_FILE = os.getenv('BAD_CASES_FILE', 'to_fix.txt')

# Paramètres d'affichage
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'fr')
DEFAULT_ALPHA = float(os.getenv('DEFAULT_ALPHA', 0.3)) 