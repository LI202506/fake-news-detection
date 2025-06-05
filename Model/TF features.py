import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import sklearn
import time
import platform
import os
import sys
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try optional imports - these won't fail if not present
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("tqdm not available, progress bars disabled")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not available, plotting disabled")

def log_system_info():
    """Record system and package information"""
    print("\n=== System Information ===")
    print(f"pandas_version: {pd.__version__}")
    print(f"numpy_version: {np.__version__}")
    print(f"sklearn_version: {sklearn.__version__}")
    print(f"python_version: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU information
    try:
        import torch
        print(f"gpu_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu_count: {torch.cuda.device_count()}")
            print(f"gpu_details: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    except ImportError:
        print("gpu_available: Unknown (torch not installed)")

def main():
    total_start_time = time.time()
    
    print("GPU available:", "Checking...")
    log_system_info()
    
    # =============================
    # 1) Load data
    # =============================
    print("\n=== 1) Load data ===")
    data_start_time = time.time()
    
    try:
        data = pd.read_csv("fake_news_all.csv")
        print("Data preview:")
        print(data.head())
        
        # Basic dataset statistics
        print(f"Dataset size: {data.shape}")
        print("Class distribution:")
        print(data["label"].value_counts())
        
        # Calculate text length statistics
        data["text_length"] = data["text"].apply(len)
        print(f"Average text length: {data['text_length'].mean():.2f} characters")
        print(f"Maximum text length: {data['text_length'].max()} characters")
        print(f"Minimum text length: {data['text_length'].min()} characters")
        
        # Merge title and text columns to generate complete text
        data["full_text"] = data["title"] + " " + data["text"]
        data.dropna(subset=["full_text", "label"], inplace=True)
        
        data_end_time = time.time()
        print(f"Data loading and preprocessing time: {data_end_time - data_start_time:.2f} seconds")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # =============================
    # 2) Split train/test sets
    # =============================
    print("\n=== 2) Split train/test sets ===")
    split_start_time = time.time()
    
    X = data["full_text"].tolist()
    y = data["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create validation set from training data for model tuning
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    # Use the split training set for better generalization
    X_train = X_train_final
    y_train = y_train_final
    
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}")
    
    # Check class distribution
    train_labels = np.array(y_train)
    val_labels = np.array(y_val)
    test_labels = np.array(y_test)
    print(f"Training set class distribution: {np.bincount(train_labels)}")
    print(f"Validation set class distribution: {np.bincount(val_labels)}")
    print(f"Test set class distribution: {np.bincount(test_labels)}")
    
    split_end_time = time.time()
    print(f"Data splitting time: {split_end_time - split_start_time:.2f} seconds")

    # =============================
    # 3) Text feature extraction: TF-IDF Vectorizer
    # =============================
    print("\n=== 3) Text feature extraction ===")
    feature_start_time = time.time()
    
    # Optimized TF-IDF configuration for balanced performance and speed
    max_features = 800  # Balanced feature count for efficiency
    print(f"Using TF-IDF feature extraction, max features: {max_features}")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),  # Focus on unigrams for computational efficiency
        min_df=3,           # Remove rare terms for better generalization
        max_df=0.85,        # Remove very common terms
        stop_words='english'  # Remove English stop words
    )
    
    print("Extracting features for training set...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    print("Extracting features for validation set...")
    X_val_tfidf = vectorizer.transform(X_val)
    
    print("Extracting features for test set...")
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature dimensions: {X_train_tfidf.shape}")
    print(f"Feature sparsity: {X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]):.4f}")
    
    # Record top features
    feature_names = vectorizer.get_feature_names_out()
    if len(feature_names) > 10:
        print(f"Top 10 features: {feature_names[:10]}")
    
    feature_end_time = time.time()
    print(f"Feature extraction time: {feature_end_time - feature_start_time:.2f} seconds")
    print(f"Average feature extraction time per sample: {(feature_end_time - feature_start_time) * 1000 / len(X_train):.2f} milliseconds")

    # =============================
    # 4) Build and train Logistic Regression model
    # =============================
    print("\n=== 4) Build and train Logistic Regression model ===")
    model_start_time = time.time()
    
    # Conservative LR configuration for stable performance
    C_value = 0.3  # Conservative regularization for better generalization
    max_iter = 500  # Sufficient iterations while avoiding overfitting
    solver = 'liblinear'  # Reliable solver for text classification
    
    print(f"Model configuration: C={C_value}, max_iter={max_iter}, solver={solver}")
    clf = LogisticRegression(
        C=C_value, 
        max_iter=max_iter, 
        solver=solver, 
        verbose=1, 
        n_jobs=-1,
        penalty='l2',  # L2 regularization
        random_state=42
    )
    
    print("Starting training...")
    clf.fit(X_train_tfidf, y_train)
    
    # Record model information
    try:
        print(f"Model convergence iterations: {clf.n_iter_}")
    except AttributeError:
        print("Unable to get convergence information")
    
    model_end_time = time.time()
    training_time = model_end_time - model_start_time
    print(f"Model training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # =============================
    # 5) Model evaluation
    # =============================
    print("\n=== 5) Model evaluation ===")
    eval_start_time = time.time()
    
    # Calculate accuracy on training, validation, and test sets
    train_accuracy = clf.score(X_train_tfidf, y_train)
    val_accuracy = clf.score(X_val_tfidf, y_val)
    test_accuracy = clf.score(X_test_tfidf, y_test)
    
    print(f"Training set accuracy: {train_accuracy:.4f}")
    print(f"Validation set accuracy: {val_accuracy:.4f}")
    print(f"Test set accuracy: {test_accuracy:.4f}")
    
    # Check for overfitting
    overfitting_gap = train_accuracy - test_accuracy
    print(f"Overfitting gap (train - test): {overfitting_gap:.4f}")
    
    # Detailed evaluation metrics
    print("Detailed evaluation...")
    
    # Predictions and probabilities
    y_pred = clf.predict(X_test_tfidf)
    y_prob = clf.predict_proba(X_test_tfidf)[:, 1]
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Detailed classification report
    print("\nClassification Report:")
    class_report = classification_report(y_test, y_pred, digits=4)
    print(class_report)
    
    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Average inference time per sample: {eval_time * 1000 / len(X_test):.2f} milliseconds")
    
    # =============================
    # 6) Threshold optimization
    # =============================
    print("\n=== 6) Threshold optimization ===")
    threshold_start_time = time.time()
    
    # Search for optimal threshold using validation set
    thresholds = np.linspace(0.2, 0.8, 61)
    best_threshold = 0.5
    best_val_accuracy = 0
    
    # Use validation set for threshold optimization
    y_val_prob = clf.predict_proba(X_val_tfidf)[:, 1]
    
    for threshold in thresholds:
        y_val_pred_custom = (y_val_prob >= threshold).astype(int)
        val_accuracy_custom = (y_val_pred_custom == y_val).mean()
        
        if val_accuracy_custom > best_val_accuracy:
            best_val_accuracy = val_accuracy_custom
            best_threshold = threshold
    
    # Apply best threshold to test set
    y_pred_optimized = (y_prob >= best_threshold).astype(int)
    optimized_accuracy = (y_pred_optimized == y_test).mean()
    conf_matrix_optimized = confusion_matrix(y_test, y_pred_optimized)
    class_report_optimized = classification_report(y_test, y_pred_optimized, digits=4)
    
    threshold_end_time = time.time()
    print(f"Threshold optimization time: {threshold_end_time - threshold_start_time:.2f} seconds")
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Test accuracy with optimal threshold: {optimized_accuracy:.4f}")
    print("Confusion matrix with optimal threshold:")
    print(conf_matrix_optimized)
    print("Classification report with optimal threshold:")
    print(class_report_optimized)

    # =============================
    # 7) Save model and results
    # =============================
    print("\n=== 7) Save model and results ===")
    save_start_time = time.time()
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Save model
    try:
        import joblib
        model_path = "output/tfidf_lr_model.joblib"
        joblib.dump((vectorizer, clf, best_threshold), model_path)
        print(f"Model saved to: {model_path}")
    except ImportError:
        print("joblib not installed, cannot save model")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': vectorizer.get_feature_names_out(),
        'importance': clf.coef_[0]
    })
    feature_importance = feature_importance.sort_values('importance', key=abs, ascending=False)
    feature_importance.to_csv("output/feature_importance.csv", index=False)
    
    # Save comprehensive results
    results = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "TF-IDF + Logistic Regression",
        "hyperparameters": {
            "max_features": max_features,
            "C_value": C_value,
            "solver": solver,
            "max_iter": max_iter,
            "ngram_range": "(1, 1)",
            "min_df": 3,
            "max_df": 0.85
        },
        "dataset_info": {
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "total_features": X_train_tfidf.shape[1],
            "feature_sparsity": float(X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))
        },
        "performance_metrics": {
            "training_accuracy": float(train_accuracy),
            "validation_accuracy": float(val_accuracy),
            "test_accuracy": float(test_accuracy),
            "roc_auc": float(roc_auc),
            "optimized_threshold": float(best_threshold),
            "optimized_accuracy": float(optimized_accuracy),
            "overfitting_gap": float(overfitting_gap)
        },
        "confusion_matrix": conf_matrix_optimized.tolist(),
        "timing": {
            "training_time_seconds": float(training_time),
            "total_time_seconds": float(time.time() - total_start_time)
        }
    }
    
    with open("output/tfidf_lr_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    save_end_time = time.time()
    print(f"Results saved to: output/ (time: {save_end_time - save_start_time:.2f} seconds)")
    
    # =============================
    # 8) Experiment summary
    # =============================
    total_time = time.time() - total_start_time
    print("\n=== Experiment Summary ===")
    print(f"Model: TF-IDF + Logistic Regression")
    print(f"Features: {max_features} (TF-IDF unigrams)")
    print(f"Regularization: C={C_value} (conservative)")
    print(f"Training samples: {len(X_train)}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Optimized accuracy: {optimized_accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Total runtime: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()