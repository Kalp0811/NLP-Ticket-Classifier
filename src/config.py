# src/config.py

from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
BASELINE_MODEL_DIR = MODEL_DIR / "baseline"
TRANSFORMER_MODEL_DIR = MODEL_DIR / "ticket_classifier"

# Ensure model directories exist
MODEL_DIR.mkdir(exist_ok=True)
BASELINE_MODEL_DIR.mkdir(exist_ok=True)
TRANSFORMER_MODEL_DIR.mkdir(exist_ok=True)


# Data Files
FULL_DATA_PATH = DATA_DIR / "customer_support_tickets.csv"
SAMPLED_DATA_PATH = DATA_DIR / "tickets_sampled_300.csv"
SAMPLE_SIZE = 300

# Model & Training
MODEL_NAME = "distilbert-base-uncased"

# Labels
# Define the final labels 
LABELS = ["Billing", "Technical", "Other"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# Map original 'Ticket Type' values to our target labels
TICKET_TYPE_MAP = {
    'Billing inquiry': 'Billing',
    'Product inquiry': 'Other',
    'Technical issue': 'Technical'
}