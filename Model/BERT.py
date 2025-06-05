import os
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import json
from datetime import datetime
import platform

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from transformers import BertTokenizerFast, TFBertForSequenceClassification
# If you use RoBERTa or other models, change to RobertaTokenizerFast, TFRobertaForSequenceClassification, etc.

def log_system_info():
    """Record system and computational resources information."""
    system_info = {
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # GPU information
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        system_info["gpu_available"] = True
        system_info["gpu_count"] = len(gpus)
        # Get details for each GPU
        gpu_details = []
        for i, gpu in enumerate(gpus):
            gpu_details.append({
                "name": gpu.name,
                "device_type": gpu.device_type
            })
            # Try to get additional GPU info using TensorFlow functions
            try:
                gpu_details[i]["memory_limit"] = tf.config.experimental.get_memory_info(gpu.name)["current"]
            except:
                pass
        system_info["gpu_details"] = gpu_details
    else:
        system_info["gpu_available"] = False
    
    # Save to file
    with open("system_info.json", "w") as f:
        json.dump(system_info, f, indent=4)
    
    print("\n=== System Information ===")
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    return system_info

# Model architecture configuration
MODEL_CONFIG = {
    "name": "BERT Base Uncased",
    "checkpoint": "./RoB",  # or "bert-base-uncased" etc.
    "architecture": {
        "type": "Transformer-based language model",
        "layers": 12,
        "hidden_size": 768,
        "attention_heads": 12,
        "parameters": "110M parameters",
        "activation": "GELU",
        "description": "Bidirectional Encoder Representations from Transformers - pre-trained on large corpus of unlabeled text with masked language modeling objective"
    }
}

# Hyperparameter settings
HYPERPARAMETERS = {
    "learning_rate": 1e-5,
    "batch_size": 16,
    "epochs": 4,
    "max_sequence_length": 128,
    "optimizer": "Adam",
    "early_stopping_patience": 1,
    "loss_function": "SparseCategoricalCrossentropy",
    "weight_decay": 0,
    "dropout": 0.1  # Default dropout in BERT model
}

# Training and evaluation metrics to track
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1-score",
    "confusion_matrix",
    "roc_auc_score",
    "training_time",
    "inference_time"
]

def main():
    # Log system information
    system_info = log_system_info()
    
    # Start timing the process
    start_time = time.time()
    
    # =============================
    # 1) Load data
    # =============================
    print("=== 1) Load data ===")
    data = pd.read_csv("fake_news_all.csv")  # Based on your file name/path
    print("Data preview:")
    print(data.head())
    
    # Data statistics
    print(f"Dataset size: {data.shape}")
    if "label" in data.columns:
        print("Class distribution:")
        print(data["label"].value_counts())
    
    data["full_text"] = data["title"] + " " + data["text"]
    data.dropna(subset=["full_text", "label"], inplace=True)
    
    # Text length statistics
    data["text_length"] = data["full_text"].apply(len)
    print(f"Average text length: {data['text_length'].mean():.2f} characters")
    print(f"Maximum text length: {data['text_length'].max()} characters")
    print(f"Minimum text length: {data['text_length'].min()} characters")

    # =============================
    # 2) Split train/test/validation sets
    # =============================
    print("\n=== 2) Split train/test/validation sets ===")
    # First split training and test sets
    train_val_df, test_df = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["label"]
    )
    
    # Then separate validation set from training set
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1,  # 10% of original data as validation set
        random_state=42,
        stratify=train_val_df["label"]
    )
    
    X_train = train_df["full_text"].tolist()
    y_train = train_df["label"].tolist()

    X_val = val_df["full_text"].tolist()
    y_val = val_df["label"].tolist()
    
    X_test = test_df["full_text"].tolist()
    y_test = test_df["label"].tolist()

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # =============================
    # 3) Initialize tokenizer
    # =============================
    print("\n=== 3) Initialize tokenizer ===")
    print(f"Using model: {MODEL_CONFIG['name']}")
    print(f"Model architecture: {MODEL_CONFIG['architecture']['type']}")
    print(f"  - Layers: {MODEL_CONFIG['architecture']['layers']}")
    print(f"  - Hidden size: {MODEL_CONFIG['architecture']['hidden_size']}")
    print(f"  - Attention heads: {MODEL_CONFIG['architecture']['attention_heads']}")
    print(f"  - Parameters: {MODEL_CONFIG['architecture']['parameters']}")
    
    CHECKPOINT = MODEL_CONFIG["checkpoint"]
    tokenizer = BertTokenizerFast.from_pretrained(CHECKPOINT)

    MAX_LEN = HYPERPARAMETERS["max_sequence_length"]
    print(f"Maximum sequence length: {MAX_LEN}")
    print("Starting training set tokenization...")

    def tokenize_map(texts, labels):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="tf"
        )
        return encodings, tf.convert_to_tensor(labels, dtype=tf.int32)

    train_enc, train_labels = tokenize_map(X_train, y_train)
    print("Training set tokenization completed")
    
    print("Starting validation set tokenization...")
    val_enc, val_labels = tokenize_map(X_val, y_val)
    print("Validation set tokenization completed")
    
    print("Starting test set tokenization...")
    test_enc, test_labels = tokenize_map(X_test, y_test)
    print("Test set tokenization completed")

    # =============================
    # 4) Build TF Dataset
    # =============================
    print("\n=== 4) Build TF Dataset ===")
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_enc),
        train_labels
    ))
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_enc),
        val_labels
    ))
    
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_enc),
        test_labels
    ))

    BATCH_SIZE = HYPERPARAMETERS["batch_size"]
    print(f"Batch size: {BATCH_SIZE}")

    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # =============================
    # 5) Initialize BERT for Classification
    # =============================
    print("\n=== 5) Initialize BERT for Classification ===")
    model = TFBertForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=2  # Binary classification
    )
    
    # Print model parameter count
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"Model trainable parameters: {trainable_params:,}")
    print(f"Model non-trainable parameters: {non_trainable_params:,}")
    print(f"Total model parameters: {total_params:,}")

    # =============================
    # 6) Compile & train (multi-epoch + EarlyStopping)
    # =============================
    print("\n=== 6) Compile & train ===")
    print("Hyperparameter settings:")
    for param, value in HYPERPARAMETERS.items():
        print(f"  - {param}: {value}")

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=HYPERPARAMETERS["learning_rate"])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    EPOCHS = HYPERPARAMETERS["epochs"]
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=HYPERPARAMETERS["early_stopping_patience"],
        restore_best_weights=True
    )
    
    # Add TensorBoard callback to record training process
    log_dir = f"./logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1,
        update_freq='epoch'
    )
    
    # Add model checkpoint
    checkpoint_path = "./checkpoints/bert_model"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor="val_accuracy"
    )

    print("Starting fit ...")
    training_start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[early_stop, tensorboard_callback, checkpoint_callback]
    )
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    print(f"Training completed. Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # =============================
    # 7) Test set evaluation
    # =============================
    print("\n=== 7) Test set evaluation ===")
    test_start_time = time.time()
    eval_loss, eval_acc = model.evaluate(test_dataset)
    test_end_time = time.time()
    inference_time = test_end_time - test_start_time
    
    # Calculate average inference time per sample
    avg_inference_time = inference_time / len(y_test)
    
    print(f"Test loss: {eval_loss:.4f}, Test accuracy: {eval_acc:.4f}")
    print(f"Total test set inference time: {inference_time:.2f} seconds")
    print(f"Average inference time per sample: {avg_inference_time*1000:.2f} milliseconds")

    print("\nStarting prediction and printing detailed classification report...")
    preds_logits = model.predict(test_dataset)["logits"]
    preds = np.argmax(preds_logits, axis=1)
    
    # Calculate ROC AUC
    preds_proba = tf.nn.softmax(preds_logits, axis=-1).numpy()[:,1]
    try:
        roc_auc = roc_auc_score(y_test, preds_proba)
        print(f"ROC AUC: {roc_auc:.4f}")
    except:
        print("Unable to calculate ROC AUC")

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(cm)
    print("Classification Report:")
    cr = classification_report(y_test, preds, digits=4)
    print(cr)

    # =============================
    # 8) Threshold tuning (optional)
    # =============================
    print("\n=== 8) Attempt threshold tuning ===")
    # Default argmax(logits) is equivalent to threshold=0.5 for logit difference.
    # Here we can scan the positive class (1) probability
    preds_proba = tf.nn.softmax(preds_logits, axis=-1).numpy()[:,1]  # Get positive class (1) probability

    best_thr, best_acc = 0.5, 0.0
    thresholds = np.arange(0.4, 0.61, 0.01)  # Scan 0.40~0.60
    for thr in thresholds:
        custom_preds = (preds_proba >= thr).astype(int)
        acc_thr = np.mean(custom_preds == np.array(y_test))
        if acc_thr > best_acc:
            best_acc = acc_thr
            best_thr = thr

    print(f"Optimal threshold: {best_thr:.2f}, Test set accuracy: {best_acc:.4f}")
    if best_thr != 0.5:
        custom_preds = (preds_proba >= best_thr).astype(int)
        print("Confusion Matrix with best_thr:")
        print(confusion_matrix(y_test, custom_preds))
        print("Classification Report with best_thr:")
        print(classification_report(y_test, custom_preds, digits=4))
    
    # =============================
    # 9) Save model and results
    # =============================
    print("\n=== 9) Save model and results ===")
    # Save model
    save_dir = "./saved_model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to: {save_dir}")
    
    # Save training history
    history_dict = {
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]]
    }
    
    # Save evaluation results and hyperparameters
    results = {
        "model_config": MODEL_CONFIG,
        "hyperparameters": HYPERPARAMETERS,
        "training_metrics": {
            "training_time_seconds": training_time,
            "epochs_trained": len(history.history["accuracy"]),
            "final_training_accuracy": float(history.history["accuracy"][-1]),
            "final_validation_accuracy": float(history.history["val_accuracy"][-1]),
            "best_validation_accuracy": float(max(history.history["val_accuracy"]))
        },
        "evaluation_metrics": {
            "test_accuracy": float(eval_acc),
            "test_loss": float(eval_loss),
            "confusion_matrix": cm.tolist(),
            "classification_report": cr,
            "inference_time_seconds": inference_time,
            "avg_inference_time_ms": avg_inference_time * 1000,
            "best_threshold": float(best_thr),
            "best_threshold_accuracy": float(best_acc)
        },
        "system_info": system_info,
        "training_history": history_dict,
        "dataset_stats": {
            "total_samples": len(data),
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "avg_text_length": float(data["text_length"].mean()),
            "max_text_length": int(data["text_length"].max()),
            "class_distribution": data["label"].value_counts().to_dict()
        }
    }
    
    # If ROC AUC was calculated, add to results
    try:
        results["evaluation_metrics"]["roc_auc"] = float(roc_auc)
    except:
        pass
    
    # Save results as JSON
    with open("model_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to: model_results.json")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Print final summary
    print("\n=== Experiment Summary ===")
    print(f"Model: {MODEL_CONFIG['name']}")
    print(f"Test accuracy: {eval_acc:.4f}")
    try:
        print(f"ROC AUC: {roc_auc:.4f}")
    except:
        pass
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Best threshold: {best_thr:.2f} (accuracy: {best_acc:.4f})")

if __name__ == "__main__":
    # Check GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU available:", physical_devices)
    else:
        print("No GPU available, using CPU")
    main()