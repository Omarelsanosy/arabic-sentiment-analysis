
# =========================
# Imports
# =========================

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# =========================
# MLflow Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACT_DIR = PROJECT_ROOT / "mlruns"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"
MLFLOW_EXPERIMENT_NAME = "arabic-sentiment-analysis"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")


# =========================
# Device
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =========================
# Model Configuration
# =========================

ARABERT_MODEL = "aubmindlab/bert-base-arabertv2"

BERT_MAX_LEN = 64
BERT_BATCH = 16
BERT_EPOCHS = 3
BERT_LR = 2e-5

RANDOM_STATE = 42

print(f"Model: {ARABERT_MODEL}")
print(f"Max length: {BERT_MAX_LEN}")
print(f"Batch size: {BERT_BATCH}")
print(f"Learning rate: {BERT_LR}")
print(f"Epochs: {BERT_EPOCHS}")


# =========================
# Load Dataset
# =========================

DATA_PATH = "data/raw/CompanyReviews.csv"

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={'review_description': 'text', 'rating': 'sentiment'})
df = df[['text', 'sentiment', 'company']]

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print(df.head())


# =========================
# Basic EDA
# =========================

print("Dataset shape:", df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicates:")
print(df.duplicated().sum())


# =========================
# Sentiment Distribution
# =========================

label_names = {
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
}

df["sentiment_label"] = df["sentiment"].map(label_names)

print("Sentiment distribution:")
print(df["sentiment_label"].value_counts())

print("\nSentiment percentage:")
print(
    df["sentiment_label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# =========================
# Sentiment Visualization
# =========================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sentiment_counts = df["sentiment_label"].value_counts()

sentiment_counts.plot(
    kind="bar",
    ax=axes[0]
)

axes[0].set_title("Sentiment Count")
axes[0].set_xlabel("Sentiment")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0)

sentiment_counts.plot(
    kind="pie",
    ax=axes[1],
    autopct="%1.1f%%"
)

axes[1].set_title("Sentiment Distribution")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()


# =========================
# Reviews per Company
# =========================

plt.figure(figsize=(12, 5))

df["company"].value_counts().plot(
    kind="bar"
)

plt.title("Reviews per Company")
plt.xlabel("Company")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# =========================
# Basic Text Cleaning
# =========================

df = df.dropna(
    subset=["text", "sentiment"]
).copy()

print("After dropping nulls:", df.shape)


def clean_arabic_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_text"] = df["text"].apply(clean_arabic_text)

# Remove empty texts
df = df[
    df["clean_text"].str.strip().ne("")
].copy()

print("After removing empty texts:", df.shape)

print(
    df[["text", "clean_text", "sentiment_label"]].head()
)


# =========================
# Remove Duplicates
# =========================

before = len(df)

df = df.drop_duplicates(
    subset=["clean_text", "sentiment_label"]
).reset_index(drop=True)

after = len(df)

print(f"Removed duplicates: {before - after:,}")
print(f"Remaining samples: {after:,}")


# =========================
# Label Encoding
# =========================

le = LabelEncoder()

df["label"] = le.fit_transform(df["sentiment_label"])

print("Original classes:")
print(le.classes_)

print("\nLabel mapping:")

label2id = {
    str(label): int(i)
    for i, label in enumerate(le.classes_)
}

id2label = {
    int(i): str(label)
    for i, label in enumerate(le.classes_)
}

print(label2id)
print(id2label)


# =========================
# Train / Validation / Test Split
# =========================

X = df["clean_text"]
y = df["label"]

# 80% Train, 20% Temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# Split temporary into 10% Validation + 10% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"Train:      {len(X_train):,}")
print(f"Validation: {len(X_val):,}")
print(f"Test:       {len(X_test):,}")


# =========================
# Development Subset
# =========================

TRAIN_N = min(30_000, len(X_train))
VAL_N = min(4_000, len(X_val))
TEST_N = min(8_000, len(X_test))

Xb_train = X_train.iloc[:TRAIN_N].reset_index(drop=True)
yb_train = y_train.iloc[:TRAIN_N].reset_index(drop=True)

Xb_val = X_val.iloc[:VAL_N].reset_index(drop=True)
yb_val = y_val.iloc[:VAL_N].reset_index(drop=True)

Xb_test = X_test.iloc[:TEST_N].reset_index(drop=True)
yb_test = y_test.iloc[:TEST_N].reset_index(drop=True)

print(f"Train subset: {len(Xb_train):,}")
print(f"Validation subset: {len(Xb_val):,}")
print(f"Test subset: {len(Xb_test):,}")


# =========================
# Class Weights
# =========================

classes = np.unique(yb_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=yb_train
)

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float
).to(DEVICE)

print("Class weights:")

for class_id, weight in zip(classes, class_weights):
    print(
        f"{class_id} "
        f"({id2label[int(class_id)]}): "
        f"{weight:.3f}"
    )


# =========================
# Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    ARABERT_MODEL
)

print("Tokenizer loaded successfully.")


# =========================
# PyTorch Dataset
# =========================

class ArabertDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = tokenizer(
            self.texts[idx],
            max_length=BERT_MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        }


# =========================
# DataLoaders
# =========================

train_dataset = ArabertDataset(
    Xb_train,
    yb_train
)

val_dataset = ArabertDataset(
    Xb_val,
    yb_val
)

test_dataset = ArabertDataset(
    Xb_test,
    yb_test
)


bert_train_loader = DataLoader(
    train_dataset,
    batch_size=BERT_BATCH,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)

bert_val_loader = DataLoader(
    val_dataset,
    batch_size=BERT_BATCH,
    shuffle=False,
    pin_memory=torch.cuda.is_available()
)

bert_test_loader = DataLoader(
    test_dataset,
    batch_size=BERT_BATCH,
    shuffle=False,
    pin_memory=torch.cuda.is_available()
)

print("Train batches:", len(bert_train_loader))
print("Validation batches:", len(bert_val_loader))
print("Test batches:", len(bert_test_loader))


# =========================
# Load AraBERT
# =========================

NUM_CLASSES = len(le.classes_)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    ARABERT_MODEL,
    num_labels=NUM_CLASSES,
    id2label=id2label,
    label2id=label2id
)

bert_model = bert_model.to(DEVICE)

print("AraBERT model loaded.")
print("Number of classes:", NUM_CLASSES)


# =========================
# Loss / Optimizer / Scheduler
# =========================

bert_criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)

optimizer_b = AdamW(
    bert_model.parameters(),
    lr=BERT_LR,
    eps=1e-8
)

total_steps = (
    len(bert_train_loader) * BERT_EPOCHS
)

scheduler = get_linear_schedule_with_warmup(
    optimizer_b,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

print("Total training steps:", total_steps)


# =========================
# Training Function
# =========================

def bert_train_epoch(model, loader):

    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for batch in progress:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer_b.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = bert_criterion(
            outputs.logits,
            labels
        )

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer_b.step()
        scheduler.step()

        predictions = outputs.logits.argmax(dim=1)

        total_loss += loss.item()

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


# =========================
# Evaluation Function
# =========================

def bert_evaluate(model, loader):

    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        progress = tqdm(
            loader,
            desc="Evaluating",
            leave=False
        )

        for batch in progress:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = bert_criterion(
                outputs.logits,
                labels
            )

            predictions = outputs.logits.argmax(dim=1)

            total_loss += loss.item()

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return (
        avg_loss,
        accuracy,
        f1,
        all_predictions,
        all_labels
    )


# =========================
# Training
# =========================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_f1": []
}

print("Training AraBERT...\n")

with mlflow.start_run(run_name="arabert-training") as run:
    mlflow.log_params({
        "model_name": ARABERT_MODEL,
        "max_length": BERT_MAX_LEN,
        "batch_size": BERT_BATCH,
        "epochs": BERT_EPOCHS,
        "learning_rate": BERT_LR,
        "random_state": RANDOM_STATE,
        "train_samples": len(Xb_train),
        "val_samples": len(Xb_val),
        "test_samples": len(Xb_test),
        "num_classes": NUM_CLASSES,
        "device": str(DEVICE),
    })

    for epoch in range(1, BERT_EPOCHS + 1):

        train_loss, train_acc = bert_train_epoch(
            bert_model,
            bert_train_loader
        )

        val_loss, val_acc, val_f1, _, _ = bert_evaluate(
            bert_model,
            bert_val_loader
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_f1"].append(val_f1)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)

        print(
            f"Epoch {epoch}/{BERT_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )
    
    # =========================
    # Training Curves
    # =========================

    epochs = range(1, BERT_EPOCHS + 1)
    loss_plot_path = (
        PROJECT_ROOT
        / "training_curves.png"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.gcf()
    mlflow.log_figure(
        plt.gcf(),
        "training_curves.png"
    )

    plt.show()


    # =========================
    # Final Test Evaluation
    # =========================

    test_loss, test_acc, test_f1, y_pred, y_true = bert_evaluate(
        bert_model,
        bert_test_loader
    )

    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1_macro", test_f1)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Macro: {test_f1:.4f}")


    # =========================
    # Classification Report
    # =========================

    target_names = [
        id2label[i]
        for i in range(NUM_CLASSES)
    ]

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4
        )
    )
    mlflow.log_text(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4
        ),
        "classification_report.txt"
    )

    # =========================
    # Confusion Matrix
    # =========================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )

    plt.title("AraBERT Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.gcf()
    mlflow.log_figure(
        plt.gcf(),
        "confusion_matrix.png"
    )

    plt.show()

    save_path = PROJECT_ROOT / "models" / "arabert_sentiment"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bert_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    mlflow.log_artifact(
        str(save_path),
        artifact_path="arabert_sentiment"
    )


    
    mlflow.set_tag(
        "framework",
        "PyTorch"
    )


    mlflow.set_tag(
        "task",
        "Arabic Sentiment Classification"
    )


    mlflow.set_tag(
        "model_type",
        "AraBERT"
    )
print(f"MLflow run started: {run.info.run_id}")


if __name__ == "__main__":
    print("Training script completed successfully.")

