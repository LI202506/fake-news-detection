import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import json
from datetime import datetime
import platform

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Transformers
from transformers import (
    BertTokenizerFast,
    TFBertForSequenceClassification,
    create_optimizer
)

# Model architecture configuration
MODEL_CONFIG = {
    "name": "BERT Large Uncased",
    "checkpoint": "./my_bert_large_uncased/",
    "architecture": {
        "type": "Transformer-based language model (BERT Large)",
        "layers": 24,  # BERT Large layer count
        "hidden_size": 1024,  # BERT Large hidden size
        "attention_heads": 16,  # BERT Large attention heads
        "parameters": "340M parameters",  # Approximately 340 million parameters
        "activation": "GELU",
        "description": "BERT Large - Bidirectional Encoder Representations from Transformers large version"
    }
}

# Hyperparameter settings
HYPERPARAMETERS = {
    "max_sequence_length": 128,
    "batch_size": 16,
    "epochs": 6,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_proportion": 0.1,  # 10% of total training steps for warmup
    "optimizer": "AdamW",
    "early_stopping_patience": 1,
    "loss_function": "SparseCategoricalCrossentropy",
    "classification_layers": "Single linear classifier",
    "cv_folds": 5,  # New: Cross-validation folds
    "cv_random_state": 42  # New: Cross-validation random seed
}

# Training and evaluation metrics
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

def log_system_info():
    """Record system and computational resources information"""
    system_info = {
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "transformers_version": "4.x",  # Modify according to actual version
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # GPU information
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        system_info["gpu_available"] = True
        system_info["gpu_count"] = len(gpus)
        gpu_details = []
        for i, gpu in enumerate(gpus):
            gpu_details.append({
                "name": gpu.name,
                "device_type": gpu.device_type
            })
            # Try to get GPU memory information
            try:
                gpu_details[i]["memory_limit"] = tf.config.experimental.get_memory_info(gpu.name)["current"]
            except:
                pass
        system_info["gpu_details"] = gpu_details
    else:
        system_info["gpu_available"] = False
    
    # Save to file
    with open("system_info_bert_large_cv.json", "w") as f:
        json.dump(system_info, f, indent=4)
    
    print("\n=== System Information ===")
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    return system_info

def create_model_and_tokenizer(model_name):
    """Function to create model and tokenizer, used for re-initialization in each fold"""
    print("Initializing new model and tokenizer...")
    
    # Load tokenizer
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_name, local_files_only=True)
        print("Successfully loaded tokenizer from local files")
    except:
        print("Local loading failed, trying to download tokenizer from Hugging Face")
        tokenizer = BertTokenizerFast.from_pretrained(model_name, local_files_only=False)
    
    # Check if TensorFlow model file exists
    tf_model_exists = os.path.exists(os.path.join(model_name, "tf_model.h5"))
    
    if tf_model_exists:
        print("Found TensorFlow model file, loading directly...")
        try:
            model = TFBertForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=True,
                num_labels=2
            )
            print("Successfully loaded TensorFlow model from local files")
        except Exception as e:
            print(f"TensorFlow model loading failed: {str(e)}")
            raise e
    else:
        # Only PyTorch weights available, need conversion
        print("TensorFlow model file not found, converting from PyTorch weights...")
        try:
            from transformers import BertForSequenceClassification
            
            # 1. First load PyTorch model
            print("Loading PyTorch model...")
            pt_model = BertForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=True,
                num_labels=2,
                from_pt=True  # Explicitly specify loading from PyTorch
            )
            
            # 2. Save as TensorFlow format
            print("Converting and saving as TensorFlow format...")
            tf_save_directory = os.path.join(model_name, "tf_converted")
            os.makedirs(tf_save_directory, exist_ok=True)
            pt_model.save_pretrained(tf_save_directory)
            
            # 3. Load converted TensorFlow model
            print("Loading converted TensorFlow model...")
            model = TFBertForSequenceClassification.from_pretrained(
                tf_save_directory,
                local_files_only=True,
                num_labels=2
            )
            print("PyTorch → TensorFlow conversion successful")
        except Exception as e:
            print(f"Model conversion failed: {str(e)}")
            print("Trying to download pre-trained model directly from Hugging Face...")
            model = TFBertForSequenceClassification.from_pretrained(
                "bert-large-uncased",  # Use public model as fallback
                num_labels=2
            )
    
    return model, tokenizer

def encode_texts(tokenizer, texts, labels, max_len):
    """Function to encode texts"""
    encoding_start = time.time()
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="tf"
    )
    encoding_time = time.time() - encoding_start
    print(f"Encoding {len(texts)} texts took: {encoding_time:.2f} seconds, average per text: {encoding_time/len(texts)*1000:.2f} milliseconds")
    return encodings, tf.convert_to_tensor(labels, dtype=tf.int32)

def train_single_fold(model, ds_train, ds_val, fold_num, hyperparams):
    """Function to train a single fold"""
    print(f"\n--- Starting training for fold {fold_num} ---")
    
    EPOCHS = hyperparams["epochs"]
    BATCH_SIZE = hyperparams["batch_size"]
    
    # Calculate training steps
    train_samples = sum(1 for _ in ds_train.unbatch())
    steps_per_epoch = int(np.ceil(train_samples / BATCH_SIZE))
    warmup_steps = int(hyperparams["warmup_proportion"] * EPOCHS * steps_per_epoch)
    total_steps = EPOCHS * steps_per_epoch

    init_lr = hyperparams["learning_rate"]
    weight_decay = hyperparams["weight_decay"]
    
    print(f"Fold {fold_num} training steps: {total_steps}, warmup steps: {warmup_steps}")

    optimizer, schedule = create_optimizer(
        init_lr=init_lr,
        num_warmup_steps=warmup_steps,
        num_train_steps=total_steps,
        weight_decay_rate=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    # EarlyStopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=hyperparams["early_stopping_patience"],
        restore_best_weights=True
    )
    
    # Train model
    training_start_time = time.time()
    
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1
    )
    
    training_time = time.time() - training_start_time
    print(f"Fold {fold_num} training completed. Training time: {training_time:.2f} seconds")
    
    return history, training_time

def evaluate_fold(model, ds_val, y_val, fold_num):
    """Function to evaluate a single fold"""
    print(f"Evaluating fold {fold_num}...")
    
    eval_start_time = time.time()
    eval_loss, eval_acc = model.evaluate(ds_val, verbose=0)
    eval_time = time.time() - eval_start_time
    
    # Prediction
    pred_start_time = time.time()
    preds_logits = model.predict(ds_val, verbose=0)["logits"]
    pred_time = time.time() - pred_start_time
    
    preds_label = np.argmax(preds_logits, axis=1)
    
    # Calculate detailed metrics
    cm = confusion_matrix(y_val, preds_label)
    cr = classification_report(y_val, preds_label, output_dict=True)
    
    # Calculate ROC AUC
    probs = tf.nn.softmax(preds_logits, axis=-1).numpy()[:,1]
    try:
        roc_auc = roc_auc_score(y_val, probs)
    except:
        roc_auc = None
    
    fold_results = {
        "fold": fold_num,
        "accuracy": float(eval_acc),
        "loss": float(eval_loss),
        "precision": float(cr['weighted avg']['precision']),
        "recall": float(cr['weighted avg']['recall']),
        "f1_score": float(cr['weighted avg']['f1-score']),
        "roc_auc": float(roc_auc) if roc_auc else None,
        "confusion_matrix": cm.tolist(),
        "eval_time": eval_time,
        "pred_time": pred_time
    }
    
    auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    print(f"Fold {fold_num} results: Acc={eval_acc:.4f}, F1={cr['weighted avg']['f1-score']:.4f}, AUC={auc_str}")
    
    return fold_results

def main():
    # Record system information
    system_info = log_system_info()
    
    # Start timing
    total_start_time = time.time()
    
    # =============================
    # 1) Load data
    # =============================
    print("=== 1) Load data ===")
    data_loading_start = time.time()
    data = pd.read_csv("fake_news_all.csv")
    
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
    
    data_loading_time = time.time() - data_loading_start
    print(f"Data loading and preprocessing time: {data_loading_time:.2f} seconds")

    # =============================
    # 2) Data splitting - Independent test set + Cross-validation
    # =============================
    print("\n=== 2) Data splitting: Independent test set + Cross-validation ===")
    split_start_time = time.time()
    
    # First split out independent test set (20%)
    train_val_df, test_df = train_test_split(
        data,
        test_size=0.2,
        stratify=data["label"],
        random_state=42
    )
    
    # Remaining 80% data for cross-validation
    X_train_val = train_val_df["full_text"].tolist()
    y_train_val = train_val_df["label"].tolist()
    X_test = test_df["full_text"].tolist()
    y_test = test_df["label"].tolist()
    
    print(f"Train+validation set size: {len(X_train_val)}, Independent test set size: {len(X_test)}")
    print(f"Train+validation set class distribution: {np.bincount(train_val_df['label'])}")
    print(f"Independent test set class distribution: {np.bincount(test_df['label'])}")
    
    # Set up cross-validation
    CV_FOLDS = HYPERPARAMETERS["cv_folds"]
    CV_RANDOM_STATE = HYPERPARAMETERS["cv_random_state"]
    
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    print(f"Set up {CV_FOLDS}-fold stratified cross-validation")
    
    split_time = time.time() - split_start_time
    print(f"Data splitting time: {split_time:.2f} seconds")

    # =============================
    # 3) Cross-validation loop
    # =============================
    print(f"\n=== 3) {CV_FOLDS}-fold cross-validation starts ===")
    cv_start_time = time.time()
    
    MODEL_NAME = MODEL_CONFIG["checkpoint"]
    MAX_LEN = HYPERPARAMETERS["max_sequence_length"]
    BATCH_SIZE = HYPERPARAMETERS["batch_size"]
    
    # Store results for each fold
    cv_results = []
    cv_training_times = []
    cv_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/{CV_FOLDS}")
        print(f"{'='*50}")
        
        fold_start_time = time.time()
        
        # Get training and validation data for this fold
        X_train_fold = [X_train_val[i] for i in train_idx]
        y_train_fold = [y_train_val[i] for i in train_idx]
        X_val_fold = [X_train_val[i] for i in val_idx]
        y_val_fold = [y_train_val[i] for i in val_idx]
        
        print(f"Training set size: {len(X_train_fold)}, Validation set size: {len(X_val_fold)}")
        print(f"Training set class distribution: {np.bincount(y_train_fold)}")
        print(f"Validation set class distribution: {np.bincount(y_val_fold)}")
        
        # Create new model and tokenizer
        model, tokenizer = create_model_and_tokenizer(MODEL_NAME)
        
        # Encode data
        print("Encoding training data...")
        train_enc, train_labels = encode_texts(tokenizer, X_train_fold, y_train_fold, MAX_LEN)
        print("Encoding validation data...")
        val_enc, val_labels = encode_texts(tokenizer, X_val_fold, y_val_fold, MAX_LEN)
        
        # Build tf.data.Dataset
        ds_train = tf.data.Dataset.from_tensor_slices((dict(train_enc), train_labels)).shuffle(len(X_train_fold)).batch(BATCH_SIZE)
        ds_val = tf.data.Dataset.from_tensor_slices((dict(val_enc), val_labels)).batch(BATCH_SIZE)
        
        # Train model
        history, training_time = train_single_fold(model, ds_train, ds_val, fold, HYPERPARAMETERS)
        
        # Evaluate model
        fold_results = evaluate_fold(model, ds_val, y_val_fold, fold)
        
        # Save results
        cv_results.append(fold_results)
        cv_training_times.append(training_time)
        cv_histories.append({
            "accuracy": [float(x) for x in history.history["accuracy"]],
            "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
            "loss": [float(x) for x in history.history["loss"]],
            "val_loss": [float(x) for x in history.history["val_loss"]]
        })
        
        fold_time = time.time() - fold_start_time
        print(f"Fold {fold} total time: {fold_time:.2f} seconds")
        
        # Clean up memory
        del model, tokenizer, train_enc, val_enc, ds_train, ds_val
        tf.keras.backend.clear_session()
    
    cv_total_time = time.time() - cv_start_time
    print(f"\nCross-validation total time: {cv_total_time/60:.2f} minutes")

    # =============================
    # 4) Cross-validation results summary
    # =============================
    print("\n=== 4) Cross-validation results summary ===")
    
    # Calculate mean and standard deviation for each metric
    metrics_summary = {}
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    for metric in metric_names:
        values = [fold[metric] for fold in cv_results if fold[metric] is not None]
        if values:
            metrics_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values
            }
    
    print("Cross-validation results summary:")
    for metric, stats in metrics_summary.items():
        print(f"{metric}: {stats['mean']:.4f} (±{stats['std']:.4f})")
    
    # =============================
    # 5) Final evaluation on independent test set
    # =============================
    print("\n=== 5) Final evaluation on independent test set ===")
    
    # Retrain model on all training+validation data
    print("Retraining final model on all training+validation data...")
    final_model, final_tokenizer = create_model_and_tokenizer(MODEL_NAME)
    
    # Encode all training data
    train_enc_final, train_labels_final = encode_texts(final_tokenizer, X_train_val, y_train_val, MAX_LEN)
    test_enc_final, test_labels_final = encode_texts(final_tokenizer, X_test, y_test, MAX_LEN)
    
    # Build datasets
    ds_train_final = tf.data.Dataset.from_tensor_slices((dict(train_enc_final), train_labels_final)).shuffle(len(X_train_val)).batch(BATCH_SIZE)
    ds_test_final = tf.data.Dataset.from_tensor_slices((dict(test_enc_final), test_labels_final)).batch(BATCH_SIZE)
    
    # Train final model
    final_history, final_training_time = train_single_fold(final_model, ds_train_final, ds_test_final, "Final", HYPERPARAMETERS)
    
    # Final evaluation
    final_results = evaluate_fold(final_model, ds_test_final, y_test, "Final")
    
    # =============================
    # 6) Save all results
    # =============================
    print("\n=== 6) Save model and results ===")
    save_start_time = time.time()
    
    # Save final model
    save_dir = "./saved_bert_large_cv_model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    try:
        final_model.save_pretrained(save_dir)
        print(f"Final model saved to: {save_dir}")
    except Exception as e:
        print(f"Model saving error: {str(e)}")
    
    # Calculate total runtime
    total_time = time.time() - total_start_time
    
    # Collect all results
    results = {
        "model_config": MODEL_CONFIG,
        "hyperparameters": HYPERPARAMETERS,
        "system_info": system_info,
        "cross_validation": {
            "folds": CV_FOLDS,
            "cv_results": cv_results,
            "cv_histories": cv_histories,
            "cv_training_times": cv_training_times,
            "metrics_summary": metrics_summary,
            "cv_total_time_seconds": cv_total_time
        },
        "final_evaluation": {
            "independent_test_results": final_results,
            "final_training_time_seconds": final_training_time,
            "final_history": {
                "accuracy": [float(x) for x in final_history.history["accuracy"]],
                "val_accuracy": [float(x) for x in final_history.history["val_accuracy"]],
                "loss": [float(x) for x in final_history.history["loss"]],
                "val_loss": [float(x) for x in final_history.history["val_loss"]]
            }
        },
        "timing": {
            "data_loading_time_seconds": data_loading_time,
            "total_runtime_seconds": total_time,
            "total_runtime_minutes": total_time / 60
        },
        "dataset_stats": {
            "total_samples": len(data),
            "train_val_samples": len(X_train_val),
            "test_samples": len(X_test),
            "avg_text_length": float(data["text_length"].mean()),
            "max_text_length": int(data["text_length"].max()),
            "class_distribution": data["label"].value_counts().to_dict()
        }
    }
    
    # Save results as JSON
    results_path = "bert_large_cv_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    save_time = time.time() - save_start_time
    print(f"Results saved to: {results_path} (time: {save_time:.2f} seconds)")
    
    # Print final summary
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    print(f"Model: {MODEL_CONFIG['name']}")
    print(f"Cross-validation setup: {CV_FOLDS}-fold stratified cross-validation")
    print("\nCross-validation results:")
    for metric, stats in metrics_summary.items():
        print(f"  {metric}: {stats['mean']:.4f} (±{stats['std']:.4f})")
    print(f"\nIndependent test set results:")
    print(f"  accuracy: {final_results['accuracy']:.4f}")
    print(f"  f1_score: {final_results['f1_score']:.4f}")
    if final_results['roc_auc']:
        print(f"  roc_auc: {final_results['roc_auc']:.4f}")
    print(f"\nTotal runtime: {total_time/60:.2f} minutes")
    print(f"Average training time per fold: {np.mean(cv_training_times)/60:.2f} minutes")

if __name__ == "__main__":
    # Check GPU
    devs = tf.config.list_physical_devices("GPU")
    if devs:
        print("GPU available:", devs)
    else:
        print("No GPU available, using CPU")
    main()