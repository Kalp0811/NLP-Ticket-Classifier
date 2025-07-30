# src/train_transformer.py

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
from . import config

# DIAGNOSIS
import transformers
print(f"--- DIAGNOSIS: Transformers version actually being used: {transformers.__version__}")
print(f"--- DIAGNOSIS: Library location: {transformers.__file__}")

def tokenize_data(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

def train_transformer_model():
    """Main function to orchestrate transformer model training."""
    print("--- Training Transformer Model (Simplified Workaround) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    df = pd.read_csv(config.SAMPLED_DATA_PATH)
    df['label'] = df['label'].map(config.LABEL2ID)
    train_dataset = Dataset.from_pandas(df)
    tokenized_train_dataset = train_dataset.map(tokenize_data, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    ).to(device)

    training_args = TrainingArguments(
        output_dir=str(config.TRANSFORMER_MODEL_DIR / "results"),
        num_train_epochs=5,
        per_device_train_batch_size=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
    )

    print("🚀 Starting model training...")
    trainer.train()
    print("✅ Training complete!")

    print(f"💾 Saving final model and tokenizer to {config.TRANSFORMER_MODEL_DIR}...")
    model.save_pretrained(str(config.TRANSFORMER_MODEL_DIR))
    tokenizer.save_pretrained(str(config.TRANSFORMER_MODEL_DIR))
    print("✅ Model and tokenizer saved successfully!")


if __name__ == "__main__":
    train_transformer_model()