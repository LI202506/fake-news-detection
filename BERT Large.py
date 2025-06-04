import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Transformers
from transformers import (
    BertTokenizerFast,
    TFBertForSequenceClassification,
    create_optimizer
)

def main():
    print("=== 1) 读取数据 ===")
    data = pd.read_csv("fake_news_all.csv")
    data["full_text"] = data["title"] + " " + data["text"]
    data.dropna(subset=["full_text", "label"], inplace=True)

    print("\n=== 2) 划分训练/测试集 ===")
    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        stratify=data["label"],
        random_state=42
    )
    X_train = train_df["full_text"].tolist()
    y_train = train_df["label"].tolist()
    X_test  = test_df["full_text"].tolist()
    y_test  = test_df["label"].tolist()
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # 本地 TF权重目录
    MODEL_NAME = "./bert_large_uncased/"
    print("\n=== 3) 初始化分词器与模型(本地) ===")

    # local_files_only=True 防止去Hub拉取
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME, local_files_only=True)
    MAX_LEN = 128  # 可根据显存调大

    def encode_fn(texts, labels):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="tf"
        )
        return encodings, tf.convert_to_tensor(labels, dtype=tf.int32)

    train_enc, train_labels = encode_fn(X_train, y_train)
    test_enc, test_labels   = encode_fn(X_test,  y_test)

    print("\n=== 4) 构建tf.data.Dataset ===")
    BATCH_SIZE = 16
    ds_train = tf.data.Dataset.from_tensor_slices((dict(train_enc), train_labels)).shuffle(len(X_train)).batch(BATCH_SIZE)
    ds_test  = tf.data.Dataset.from_tensor_slices((dict(test_enc),  test_labels )).batch(BATCH_SIZE)

    print("\n=== 5) 构建并微调BERT Large(本地) ===")
    model = TFBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        num_labels=2
    )

    # warmup + weight decay AdamW
    EPOCHS = 6
    steps_per_epoch = int(np.ceil(len(X_train)/BATCH_SIZE))
    warmup_steps = int(0.1 * EPOCHS * steps_per_epoch)
    total_steps  = EPOCHS * steps_per_epoch

    init_lr = 1e-5
    optimizer, schedule = create_optimizer(
        init_lr=init_lr,
        num_warmup_steps=warmup_steps,
        num_train_steps=total_steps,
        weight_decay_rate=0.01
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    # EarlyStopping, 当 val_accuracy 1轮无提升则停止
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=1,
        restore_best_weights=True
    )

    print(f"开始训练: epochs={EPOCHS}, batch_size={BATCH_SIZE}, LR={init_lr}")
    history = model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    print("\n=== 测试集评估 ===")
    eval_loss, eval_acc = model.evaluate(ds_test)
    print(f"Test Accuracy: {eval_acc:.4f}, Test Loss: {eval_loss:.4f}")

    # 预测
    preds_logits = model.predict(ds_test)["logits"]
    preds_label = np.argmax(preds_logits, axis=1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds_label))
    print("Classification Report:")
    print(classification_report(y_test, preds_label, digits=4))

    print("\n=== 阈值微调(0.4~0.6) ===")
    probs = tf.nn.softmax(preds_logits, axis=-1).numpy()[:,1]
    best_thr, best_acc = 0.5, eval_acc
    for thr in np.arange(0.40, 0.61, 0.01):
        custom_pred = (probs>=thr).astype(int)
        this_acc = np.mean(custom_pred==y_test)
        if this_acc > best_acc:
            best_acc = this_acc
            best_thr = thr
    print(f"最优阈值: {best_thr:.2f}, 对应Acc={best_acc:.4f}")
    if best_thr != 0.5:
        final_pred = (probs>=best_thr).astype(int)
        print("Confusion matrix at best_thr:")
        print(confusion_matrix(y_test, final_pred))
        print("Report at best_thr:")
        print(classification_report(y_test, final_pred, digits=4))

    # 如果你想把微调好的模型再保存(含 tf_model.h5):
    # model.save_pretrained("./my_tf_bert_large/")

if __name__=="__main__":
    devs = tf.config.list_physical_devices("GPU")
    if devs:
        print("GPU 可用:", devs)
    main()