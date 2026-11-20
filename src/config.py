from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "plantvillage dataset" / "color"
MODEL_PATH = ROOT / "models" / "model.keras"
CLASSES_PATH = ROOT / "models" / "classes.json"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
