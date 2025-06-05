import os
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import json
from datetime import datetime
import platform
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

# Model architecture configuration
MODEL_CONFIG = {
    "name": "TextCNN",
    "architecture": {
        "type": "Convolutional Neural Network for text classification",
        "embedding_dim": 100,
        "filter_sizes": [3, 4, 5],  # Multi-scale convolution kernels
        "num_filters": 128,  # Number of convolution kernels for each size
        "dropout_rate": 0.5,
        "activation": "sigmoid",
        "description": "TextCNN uses multiple convolution kernels of different sizes to capture n-gram features of text, extracting the most significant features through max pooling"
    }
}

# Hyperparameter settings
HYPERPARAMETERS = {
    "max_sequence_length": 128,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 0.001,  # Adam default learning rate
    "optimizer": "adam",
    "loss_function": "binary_crossentropy",
    "early_stopping_patience": 2,  # Early stopping patience value
    "validation_split": 0.2  # Validation set ratio
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
    with open("system_info_textcnn.json", "w") as f:
        json.dump(system_info, f, indent=4)
    
    print("\n=== System Information ===")
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    return system_info

def build_textcnn_model(vocab_size, max_len, embedding_dim, filter_sizes, num_filters, dropout_rate=0.5):
    """Build TextCNN model"""
    print(f"Building TextCNN model...")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Sequence length: {max_len}")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Convolution kernel sizes: {filter_sizes}")
    print(f"  - Number of kernels per size: {num_filters}")
    print(f"  - Dropout rate: {dropout_rate}")
    
    # Define input layer
    inputs = Input(shape=(max_len,))
    
    # Embedding layer
    embedding = Embedding(input_dim=vocab_size, 
                          output_dim=embedding_dim, 
                          input_length=max_len)(inputs)
    
    # Multi-scale convolution layers
    conv_blocks = []
    for size in filter_sizes:
        conv = Conv1D(filters=num_filters, 
                      kernel_size=size, 
                      activation="relu")(embedding)
        pool = GlobalMaxPooling1D()(conv)
        conv_blocks.append(pool)
    
    # Merge results from different convolution kernels
    if len(conv_blocks) > 1:
        concat = Concatenate()(conv_blocks)
    else:
        concat = conv_blocks[0]
    
    # Dropout layer to prevent overfitting
    dropout = Dropout(dropout_rate)(concat)
    
    # Output layer
    outputs = Dense(1, activation="sigmoid")(dropout)  # Binary classification task
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def plot_training_history(history, save_path="textcnn_training_history.png"):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss curve
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history chart saved to: {save_path}")

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
    
    data = pd.read_csv("fake_news_all.csv")  # Please ensure the data file path is correct
    print("Data preview:")
    print(data.head())
    
    # Data statistics
    print(f"Dataset size: {data.shape}")
    if "label" in data.columns:
        print("Class distribution:")
        print(data["label"].value_counts())
    
    # Merge title and text columns to generate complete text
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
    # 2) Split train/validation/test sets
    # =============================
    print("\n=== 2) Split train/validation/test sets ===")
    split_start_time = time.time()
    
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
    y_train = train_df["label"].tolist()
    
    X_val = val_df["full_text"].tolist()
    y_val = val_df["label"].tolist()
    
    X_test = test_df["full_text"].tolist()
    y_test = test_df["label"].tolist()
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training set class distribution: {np.bincount(train_df['label'])}")
    print(f"Validation set class distribution: {np.bincount(val_df['label'])}")
    print(f"Test set class distribution: {np.bincount(test_df['label'])}")
    
    split_time = time.time() - split_start_time
    print(f"Data splitting time: {split_time:.2f} seconds")

    # =============================
    # 3) Text preprocessing: Using Keras Tokenizer
    # =============================
    print("\n=== 3) Text preprocessing ===")
    preprocessing_start_time = time.time()
    
    max_len = HYPERPARAMETERS["max_sequence_length"]
    print(f"Maximum sequence length: {max_len}")
    
    # Create and train tokenizer
    tokenizer = Tokenizer()
    tokenizing_start = time.time()
    tokenizer.fit_on_texts(X_train)
    tokenizing_time = time.time() - tokenizing_start
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    print(f"Tokenizer training time: {tokenizing_time:.2f} seconds")
    
    # Sequentialize and pad text
    sequence_start = time.time()
    
    print("Sequentializing training set...")
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
    
    print("Sequentializing validation set...")
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding="post", truncating="post")
    
    print("Sequentializing test set...")
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")
    
    sequence_time = time.time() - sequence_start
    print(f"Sequentialization and padding time: {sequence_time:.2f} seconds")
    
    preprocessing_time = time.time() - preprocessing_start_time
    print(f"Total text preprocessing time: {preprocessing_time:.2f} seconds")
    
    # Sequentialization result statistics
    print(f"Training set shape: {X_train_pad.shape}")
    print(f"Validation set shape: {X_val_pad.shape}")
    print(f"Test set shape: {X_test_pad.shape}")

    # =============================
    # 4) Build TextCNN model
    # =============================
    print("\n=== 4) Build TextCNN model ===")
    model_start_time = time.time()
    
    # Get model parameters from configuration
    embedding_dim = MODEL_CONFIG["architecture"]["embedding_dim"]
    filter_sizes = MODEL_CONFIG["architecture"]["filter_sizes"]
    num_filters = MODEL_CONFIG["architecture"]["num_filters"]
    dropout_rate = MODEL_CONFIG["architecture"]["dropout_rate"]
    
    model = build_textcnn_model(
        vocab_size=vocab_size,
        max_len=max_len,
        embedding_dim=embedding_dim,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        dropout_rate=dropout_rate
    )
    
    # Display model parameter count
    model.summary()
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"Model trainable parameters: {trainable_params:,}")
    print(f"Model non-trainable parameters: {non_trainable_params:,}")
    print(f"Total model parameters: {total_params:,}")
    
    # Configure model
    lr = HYPERPARAMETERS["learning_rate"]
    optimizer = HYPERPARAMETERS["optimizer"]
    loss = HYPERPARAMETERS["loss_function"]
    
    if optimizer.lower() == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = optimizer
        
    model.compile(
        loss=loss,
        optimizer=opt,
        metrics=["accuracy"]
    )
    
    model_time = time.time() - model_start_time
    print(f"Model building time: {model_time:.2f} seconds")

    # =============================
    # 5) Train model
    # =============================
    print("\n=== 5) Train model ===")
    training_start_time = time.time()
    
    batch_size = HYPERPARAMETERS["batch_size"]
    epochs = HYPERPARAMETERS["epochs"]
    patience = HYPERPARAMETERS["early_stopping_patience"]
    
    print(f"Training parameters:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Maximum training epochs: {epochs}")
    print(f"  - Early stopping patience: {patience}")
    print(f"  - Optimizer: {optimizer}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Loss function: {loss}")
    
    # Define callback functions
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "textcnn_best_model.h5"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(model_checkpoint)
    
    # TensorBoard
    log_dir = f"./logs/textcnn_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    callbacks.append(tensorboard)
    
    # Start training
    print(f"Starting training...")
    history = model.fit(
        X_train_pad,
        np.array(y_train),
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_pad, np.array(y_val)),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - training_start_time
    print(f"Model training completed, time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Plot training history
    plot_training_history(history.history)
    
    # Get actual training epochs
    actual_epochs = len(history.history['accuracy'])
    print(f"Actual training epochs: {actual_epochs}")

    # =============================
    # 6) Model evaluation and prediction
    # =============================
    print("\n=== 6) Model evaluation and prediction ===")
    evaluation_start_time = time.time()
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    loss, acc = model.evaluate(X_test_pad, np.array(y_test), verbose=1)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    
    # Prediction
    print("Generating prediction results...")
    inference_start_time = time.time()
    preds_prob = model.predict(X_test_pad)
    inference_time = time.time() - inference_start_time
    
    # Calculate inference time per sample
    avg_inference_time = inference_time / len(y_test)
    print(f"Total inference time: {inference_time:.2f} seconds")
    print(f"Average inference time per sample: {avg_inference_time*1000:.2f} milliseconds")
    
    # Default threshold (0.5) results
    preds = (preds_prob > 0.5).astype("int32").flatten()
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Classification report
    cr = classification_report(y_test, preds, digits=4)
    print("Classification Report:")
    print(cr)
    
    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, preds_prob)
        print(f"ROC AUC Score: {roc_auc:.4f}")
    except:
        roc_auc = None
        print("Unable to calculate ROC AUC")
    
    evaluation_time = time.time() - evaluation_start_time
    print(f"Evaluation time: {evaluation_time:.2f} seconds")

    # =============================
    # 7) Threshold optimization
    # =============================
    print("\n=== 7) Threshold optimization ===")
    threshold_start_time = time.time()
    
    # Try different thresholds
    thresholds = np.arange(0.3, 0.8, 0.01)
    best_threshold = 0.5
    best_accuracy = acc
    threshold_results = []
    
    for threshold in thresholds:
        threshold_preds = (preds_prob > threshold).astype("int32").flatten()
        threshold_acc = np.mean(threshold_preds == y_test)
        threshold_results.append({
            "threshold": threshold,
            "accuracy": threshold_acc
        })
        
        if threshold_acc > best_accuracy:
            best_accuracy = threshold_acc
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.2f}, Accuracy: {best_accuracy:.4f}")
    
    # Results with best threshold
    if best_threshold != 0.5:
        best_preds = (preds_prob > best_threshold).astype("int32").flatten()
        best_cm = confusion_matrix(y_test, best_preds)
        best_cr = classification_report(y_test, best_preds, digits=4)
        
        print("Confusion matrix with best threshold:")
        print(best_cm)
        print("Classification report with best threshold:")
        print(best_cr)
    else:
        best_cm = cm
        best_cr = cr
    
    threshold_time = time.time() - threshold_start_time
    print(f"Threshold optimization time: {threshold_time:.2f} seconds")

    # =============================
    # 8) Save model and results
    # =============================
    print("\n=== 8) Save model and results ===")
    save_start_time = time.time()
    
    # Save model
    model_save_path = "textcnn_model.h5"
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Save tokenizer
    import pickle
    tokenizer_path = "textcnn_tokenizer.pickle"
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to: {tokenizer_path}")
    
    # Calculate total runtime
    total_time = time.time() - total_start_time
    
    # Collect all results
    results = {
        "model_config": MODEL_CONFIG,
        "hyperparameters": HYPERPARAMETERS,
        "system_info": system_info,
        "training_stats": {
            "data_loading_time_seconds": data_loading_time,
            "preprocessing_time_seconds": preprocessing_time,
            "model_build_time_seconds": model_time,
            "training_time_seconds": training_time,
            "actual_epochs": actual_epochs,
            "total_runtime_seconds": total_time,
            "history": {k: [float(x) for x in v] for k, v in history.history.items()}
        },
        "evaluation_metrics": {
            "accuracy": float(acc),
            "loss": float(loss),
            "confusion_matrix": cm.tolist(),
            "classification_report": cr,
            "roc_auc_score": float(roc_auc) if roc_auc else None,
            "inference_time_seconds": inference_time,
            "avg_inference_time_ms": float(avg_inference_time * 1000),
            "best_threshold": float(best_threshold),
            "best_threshold_accuracy": float(best_accuracy),
            "best_threshold_confusion_matrix": best_cm.tolist(),
        },
        "model_info": {
            "trainable_parameters": int(trainable_params),
            "non_trainable_parameters": int(non_trainable_params),
            "total_parameters": int(total_params),
            "vocab_size": vocab_size
        },
        "dataset_stats": {
            "total_samples": len(data),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "avg_text_length": float(data["text_length"].mean()),
            "max_text_length": int(data["text_length"].max()),
            "min_text_length": int(data["text_length"].min()),
            "class_distribution": data["label"].value_counts().to_dict()
        },
        "threshold_optimization": {
            "thresholds_tested": [float(t) for t in thresholds],
            "results": threshold_results
        }
    }
    
    # Save results as JSON
    results_path = "textcnn_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    save_time = time.time() - save_start_time
    print(f"Results saved to: {results_path} (time: {save_time:.2f} seconds)")
    
    # Print final summary
    print("\n=== Experiment Summary ===")
    print(f"Model: {MODEL_CONFIG['name']}")
    print(f"Test accuracy: {acc:.4f}")
    if roc_auc:
        print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Best threshold: {best_threshold:.2f} (accuracy: {best_accuracy:.4f})")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Model parameters: {total_params:,}")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Total runtime: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    # If GPU is available, show GPU information
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU available:", physical_devices)
    else:
        print("No GPU available, using CPU")
    main()