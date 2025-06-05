import os
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
import platform

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve

# Transformers
from transformers import BertTokenizerFast, TFBertModel
# xgboost
import xgboost as xgb
from xgboost.callback import EarlyStopping  # Import EarlyStopping callback

# Model architecture configuration
MODEL_CONFIG = {
    "bert": {
        "name": "BERT Base Uncased",
        "checkpoint": "./RoB",  # or "bert-base-uncased"
        "architecture": {
            "type": "Transformer-based language model",
            "layers": 12,
            "hidden_size": 768,
            "attention_heads": 12,
            "parameters": "110M parameters",
            "activation": "GELU",
            "description": "Bidirectional Encoder Representations from Transformers - pre-trained bidirectional Transformer encoder"
        },
        "feature_extraction": "CLS token embedding (768-dimensional vector)"
    },
    "xgboost": {
        "name": "XGBoost Classifier",
        "architecture": {
            "type": "Gradient Boosting Decision Trees",
            "description": "An ensemble decision tree algorithm based on gradient boosting, efficiently handling high-dimensional features"
        }
    }
}

# Hyperparameter settings
HYPERPARAMETERS = {
    "bert": {
        "max_sequence_length": 128,
        "batch_size": 16,  # Increase batch size to accelerate processing
        "trainable": False,  # Whether BERT layers are trainable
        "embedding_dim": 768
    },
    "xgboost": {
        "n_estimators": 200,        # Number of trees
        "max_depth": 6,             # Maximum depth of trees
        "learning_rate": 1e-2,      # Learning rate
        "subsample": 0.8,           # Sample sampling ratio
        "colsample_bytree": 1.0,    # Feature sampling ratio
        "min_child_weight": 1,      # Minimum weight of leaf nodes
        "gamma": 0,                 # Minimum loss reduction required for splitting
        "random_state": 42,
        "eval_metric": "logloss",   # Evaluation metric
        "use_label_encoder": False,
        "early_stopping_rounds": 10  # Early stopping rounds
    }
}

# Training and evaluation metrics
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1-score",
    "confusion_matrix",
    "roc_auc_score",
    "feature_importance",
    "training_time",
    "inference_time"
]

def log_system_info():
    """Record system and computational resources information"""
    system_info = {
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "xgboost_version": xgb.__version__,
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
    with open("system_info.json", "w") as f:
        json.dump(system_info, f, indent=4)
    
    print("\n=== System Information ===")
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    return system_info

def main():
    # Record system information
    system_info = log_system_info()
    
    # Start timing
    start_time = time.time()
    
    # =============================
    # 1) Load data
    # =============================
    print("=== 1) Load data ===")
    data = pd.read_csv("fake_news_all.csv")  # Modify to your file path
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
    # 2) Split train/validation/test sets
    # =============================
    print("\n=== 2) Split train/validation/test sets ===")
    # First split training+validation set and test set
    train_val_df, test_df = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["label"]
    )
    
    # Then split training set and validation set
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1,  # 10% of original data as validation set
        random_state=42,
        stratify=train_val_df["label"]
    )
    
    X_train = train_df["full_text"].tolist()
    y_train = train_df["label"].values
    
    X_val = val_df["full_text"].tolist()
    y_val = val_df["label"].values
    
    X_test = test_df["full_text"].tolist()
    y_test = test_df["label"].values

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # =============================
    # 3) Prepare BERT tokenizer
    # =============================
    print("\n=== 3) Initialize tokenizer ===")
    print(f"Using model: {MODEL_CONFIG['bert']['name']}")
    print(f"Model architecture: {MODEL_CONFIG['bert']['architecture']['type']}")
    print(f"  - Layers: {MODEL_CONFIG['bert']['architecture']['layers']}")
    print(f"  - Hidden size: {MODEL_CONFIG['bert']['architecture']['hidden_size']}")
    print(f"  - Attention heads: {MODEL_CONFIG['bert']['architecture']['attention_heads']}")
    print(f"  - Parameters: {MODEL_CONFIG['bert']['architecture']['parameters']}")
    print(f"Feature extraction method: {MODEL_CONFIG['bert']['feature_extraction']}")
    
    CHECKPOINT = MODEL_CONFIG['bert']['checkpoint']
    tokenizer = BertTokenizerFast.from_pretrained(CHECKPOINT)
    MAX_LEN = HYPERPARAMETERS['bert']['max_sequence_length']
    print(f"Maximum sequence length: {MAX_LEN}")

    def encode_texts(texts):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="tf"
        )
        return encodings

    # =============================
    # 4) BERT model (without classification head) to extract CLS vectors
    # =============================
    print("\n=== 4) Load TFBertModel for feature extraction ===")
    bert_load_start = time.time()
    bert_model = TFBertModel.from_pretrained(CHECKPOINT)
    bert_load_time = time.time() - bert_load_start
    print(f"BERT model loading time: {bert_load_time:.2f} seconds")
    
    # Display model parameter count
    trainable_params = np.sum([np.prod(v.get_shape()) for v in bert_model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in bert_model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"BERT model trainable parameters: {trainable_params:,}")
    print(f"BERT model non-trainable parameters: {non_trainable_params:,}")
    print(f"BERT model total parameters: {total_params:,}")

    # freeze / unfreeze
    if not HYPERPARAMETERS['bert']['trainable']:
        print("BERT layers frozen (non-trainable)")
        for layer in bert_model.layers:
            layer.trainable = False
    else:
        print("BERT layers unfrozen (trainable)")

    # =============================
    # 5) Convert train/test text --> BERT CLS vectors
    # =============================
    print("\n=== 5) Extract BERT CLS vector features ===")
    batch_size = HYPERPARAMETERS['bert']['batch_size']
    print(f"Batch size: {batch_size}")

    def get_bert_embeddings(texts):
        all_embeddings = []
        start_time = time.time()
        # Process in batches to avoid GPU memory overflow
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i+batch_size]
            enc = encode_texts(batch_texts)
            # Input to bert_model
            outputs = bert_model(enc)
            # outputs.last_hidden_state shape: [batch, seq_len, hidden_dim=768]
            # CLS vector is at outputs.last_hidden_state[:,0,:]
            cls_vec = outputs.last_hidden_state[:, 0, :].numpy()  # (batch,768)
            all_embeddings.append(cls_vec)
            
            # Print progress
            if (i // batch_size) % 20 == 0:
                print(f"Processing batch {i//batch_size + 1}/{len(texts)//batch_size + 1}", end="\r")
                
        processing_time = time.time() - start_time
        print(f"\nProcessing completed, time: {processing_time:.2f} seconds, average per sample: {processing_time/len(texts)*1000:.2f} milliseconds")
        return np.concatenate(all_embeddings, axis=0), processing_time

    print("Starting training set CLS vector extraction...")
    X_train_emb, train_embedding_time = get_bert_embeddings(X_train)
    print("Training set CLS vector size:", X_train_emb.shape)
    
    print("Starting validation set CLS vector extraction...")
    X_val_emb, val_embedding_time = get_bert_embeddings(X_val)
    print("Validation set CLS vector size:", X_val_emb.shape)
    
    print("Starting test set CLS vector extraction...")
    X_test_emb, test_embedding_time = get_bert_embeddings(X_test)
    print("Test set CLS vector size:", X_test_emb.shape)

    # =============================
    # 6) Use XGBoost to train CLS vectors -> labels
    # =============================
    print("\n=== 6) XGBoost training ===")
    print(f"Using model: {MODEL_CONFIG['xgboost']['name']}")
    print(f"Model architecture: {MODEL_CONFIG['xgboost']['architecture']['type']}")
    print(f"  - Description: {MODEL_CONFIG['xgboost']['architecture']['description']}")
    
    print("XGBoost hyperparameters:")
    for param, value in HYPERPARAMETERS['xgboost'].items():
        print(f"  - {param}: {value}")
    
    # Create XGBoost classifier
    xgb_clf = xgb.XGBClassifier(
        n_estimators=HYPERPARAMETERS['xgboost']['n_estimators'],
        max_depth=HYPERPARAMETERS['xgboost']['max_depth'],
        learning_rate=HYPERPARAMETERS['xgboost']['learning_rate'],
        subsample=HYPERPARAMETERS['xgboost']['subsample'],
        colsample_bytree=HYPERPARAMETERS['xgboost']['colsample_bytree'],
        min_child_weight=HYPERPARAMETERS['xgboost']['min_child_weight'],
        gamma=HYPERPARAMETERS['xgboost']['gamma'],
        random_state=HYPERPARAMETERS['xgboost']['random_state'],
        use_label_encoder=HYPERPARAMETERS['xgboost']['use_label_encoder'],
        eval_metric=HYPERPARAMETERS['xgboost']['eval_metric']
    )
    
    # Train XGBoost model
    print("Starting XGBoost model training...")
    xgb_train_start = time.time()
    
    # Set validation set
    eval_set = [(X_train_emb, y_train), (X_val_emb, y_val)]
    
    # Create early stopping callback - compatible with XGBoost 3.0.0
    try:
        # Try using new version XGBoost callback API
        early_stopping = EarlyStopping(
            rounds=HYPERPARAMETERS['xgboost']['early_stopping_rounds'],
            save_best=True,
            metric_name='logloss' if HYPERPARAMETERS['xgboost']['eval_metric'] == 'logloss' else None
        )
        
        # Train model (using callback API)
        xgb_clf.fit(
            X_train_emb, 
            y_train,
            eval_set=eval_set,
            verbose=True,
            callbacks=[early_stopping]
        )
    except (TypeError, ImportError, AttributeError) as e:
        # If callback API is not available, try different approach
        print(f"Warning: Early stopping callback API not available ({str(e)}), trying direct training...")
        
        try:
            # Try using simple version without early stopping
            xgb_clf.fit(
                X_train_emb,
                y_train,
                eval_set=eval_set,
                verbose=True
            )
        except Exception as e2:
            print(f"Basic training also failed: {str(e2)}, trying simplest version...")
            
            # If still fails, use simplest version
            xgb_clf.fit(X_train_emb, y_train)
    
    xgb_train_time = time.time() - xgb_train_start
    print(f"XGBoost training completed, time: {xgb_train_time:.2f} seconds")
    
    # Get best iteration count (if available)
    best_iteration = None
    if hasattr(xgb_clf, 'best_iteration'):
        best_iteration = xgb_clf.best_iteration
        print(f"Best iteration: {best_iteration}")
    elif hasattr(xgb_clf, 'best_ntree_limit'):
        best_iteration = xgb_clf.best_ntree_limit
        print(f"Best iteration: {best_iteration}")
    else:
        print("Unable to get best iteration information")
        best_iteration = xgb_clf.n_estimators
    
    # =============================
    # 7) Model evaluation
    # =============================
    print("\n=== 7) Model evaluation ===")
    
    # Test set prediction
    test_pred_start = time.time()
    
    # Try using best iteration for prediction (if available)
    try:
        if best_iteration:
            y_pred = xgb_clf.predict(X_test_emb, iteration_range=(0, best_iteration))
            y_proba = xgb_clf.predict_proba(X_test_emb, iteration_range=(0, best_iteration))[:,1]
        else:
            y_pred = xgb_clf.predict(X_test_emb)
            y_proba = xgb_clf.predict_proba(X_test_emb)[:,1]
    except TypeError:
        # If iteration_range parameter is not supported
        y_pred = xgb_clf.predict(X_test_emb)
        y_proba = xgb_clf.predict_proba(X_test_emb)[:,1]
        
    test_pred_time = time.time() - test_pred_start
    
    # Calculate average inference time per sample
    avg_inference_time = test_pred_time / len(y_test)
    print(f"Total test set inference time: {test_pred_time:.2f} seconds")
    print(f"Average inference time per sample: {avg_inference_time*1000:.2f} milliseconds")
    
    # Calculate accuracy
    acc = np.mean(y_pred == y_test)
    print(f"XGBoost Accuracy: {acc:.4f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate classification report
    cr = classification_report(y_test, y_pred, digits=4)
    print("Classification Report:")
    print(cr)
    
    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")
    except:
        roc_auc = None
        print("Unable to calculate ROC AUC")
    
    # =============================
    # 8) Threshold optimization (optional)
    # =============================
    print("\n=== 8) Threshold optimization ===")
    
    best_thr, best_acc = 0.5, acc
    thresholds = np.arange(0.3, 0.8, 0.01)
    
    for thr in thresholds:
        custom_preds = (y_proba >= thr).astype(int)
        acc_thr = np.mean(custom_preds == y_test)
        if acc_thr > best_acc:
            best_acc = acc_thr
            best_thr = thr
    
    print(f"Optimal threshold: {best_thr:.2f}, Test set accuracy: {best_acc:.4f}")
    
    if best_thr != 0.5:
        custom_preds = (y_proba >= best_thr).astype(int)
        custom_cm = confusion_matrix(y_test, custom_preds)
        custom_cr = classification_report(y_test, custom_preds, digits=4)
        
        print("Confusion matrix with optimal threshold:")
        print(custom_cm)
        print("Classification report with optimal threshold:")
        print(custom_cr)
    
    # =============================
    # 9) Feature importance analysis
    # =============================
    print("\n=== 9) Feature importance analysis ===")
    
    # Get feature importance
    feature_importance = xgb_clf.feature_importances_
    
    # Calculate average feature importance
    avg_importance = np.mean(feature_importance)
    max_importance = np.max(feature_importance)
    min_importance = np.min(feature_importance)
    std_importance = np.std(feature_importance)
    
    print(f"Feature importance statistics:")
    print(f"  - Mean: {avg_importance:.6f}")
    print(f"  - Max: {max_importance:.6f}")
    print(f"  - Min: {min_importance:.6f}")
    print(f"  - Std: {std_importance:.6f}")
    
    # Find top 10 most important features
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    print("\nTop 10 most important feature indices and their importance values:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Feature #{idx}: {feature_importance[idx]:.6f}")
    
    # =============================
    # 10) Save model and results
    # =============================
    print("\n=== 10) Save model and results ===")
    
    # Save XGBoost model
    model_dir = "./saved_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    xgb_model_path = os.path.join(model_dir, "xgboost_fake_news_classifier.json")
    try:
        xgb_clf.save_model(xgb_model_path)
        print(f"XGBoost model saved to: {xgb_model_path}")
    except Exception as e:
        print(f"Unable to save model: {str(e)}")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    
    # Collect all results
    results = {
        "model_config": MODEL_CONFIG,
        "hyperparameters": HYPERPARAMETERS,
        "system_info": system_info,
        "training_stats": {
            "bert_load_time_seconds": bert_load_time,
            "bert_feature_extraction_time_seconds": {
                "train": train_embedding_time,
                "val": val_embedding_time,
                "test": test_embedding_time
            },
            "xgboost_training_time_seconds": xgb_train_time,
            "best_iteration": best_iteration if best_iteration else None,
            "total_runtime_seconds": total_time
        },
        "evaluation_metrics": {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": cr,
            "roc_auc_score": float(roc_auc) if roc_auc else None,
            "inference_time_seconds": test_pred_time,
            "avg_inference_time_ms": avg_inference_time * 1000,
            "best_threshold": float(best_thr),
            "best_threshold_accuracy": float(best_acc)
        },
        "feature_importance_stats": {
            "mean": float(avg_importance),
            "max": float(max_importance),
            "min": float(min_importance),
            "std": float(std_importance),
            "top_10_indices": top_indices.tolist(),
            "top_10_values": [float(feature_importance[idx]) for idx in top_indices]
        },
        "dataset_stats": {
            "total_samples": len(data),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "avg_text_length": float(data["text_length"].mean()),
            "max_text_length": int(data["text_length"].max()),
            "class_distribution": data["label"].value_counts().to_dict()
        }
    }
    
    # Save results as JSON
    results_path = "bert_xgboost_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"Model architecture: BERT feature extraction + XGBoost classifier")
    print(f"Test accuracy: {acc:.4f}")
    if roc_auc:
        print(f"ROC AUC: {roc_auc:.4f}")
    print(f"BERT feature extraction time: {train_embedding_time + val_embedding_time + test_embedding_time:.2f} seconds")
    print(f"XGBoost training time: {xgb_train_time:.2f} seconds")
    print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    # Check GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU available:", physical_devices)
    else:
        print("No GPU available, using CPU")
    main()