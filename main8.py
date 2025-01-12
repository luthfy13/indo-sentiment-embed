# reduced complexity
# experiment:
# - embedding          : BERT only
# - classification     : BiLSTM

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from keras.layers import Input, Bidirectional, Dense, LSTM, Dropout
from keras.models import Model
from tqdm import tqdm
from keras.optimizers import Adam
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
import keras.mixed_precision as mixed_precision

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

# Load BERT model and tokenizer
bert_model_name = "indolem/indobert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
tf_bert_model = TFBertModel.from_pretrained(bert_model_name, from_pt=True)

def tokenize_and_pad(texts, max_length=128):
    """Tokenize and pad the input texts."""
    # Convert texts to list if it's a pandas Series or numpy array
    if isinstance(texts, (pd.Series, np.ndarray)):
        texts = texts.tolist()
    
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

def create_model(max_length, tf_bert_model):
    """Create the BERT + BiLSTM model."""
    # Input layers
    input_ids = Input(shape=(max_length,), dtype='int32', name='input_ids')
    attention_mask = Input(shape=(max_length,), dtype='int32', name='attention_mask')
    
    # Get BERT output
    bert_output = tf_bert_model([input_ids, attention_mask])[0]
    
    # BiLSTM layers with reduced complexity (matching ELMo setup)
    x = Bidirectional(LSTM(32, return_sequences=True))(bert_output)
    # x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(16))(x)
    # x = Dropout(0.2)(x)
    
    # Dense layers
    x = Dense(8, activation='relu')(x)
    # x = Dropout(0.1)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Create and compile model
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    
    # Configure optimizer with mixed precision
    optimizer = Adam(learning_rate=1e-5)  # Reduced learning rate to match ELMo setup
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print/plot metrics."""
    # Generate predictions with reduced batch size
    y_pred = model.predict(X_test, batch_size=8)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Print classification report
    print(classification_report(y_test, y_pred_binary))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Calculate and print AUC-ROC score
    auc_roc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC Score: {auc_roc:.4f}")

def main():
    # Load data
    print("Loading dataset...")
    dataset = pd.read_csv("../data/PRDICT-ID/vw_dataset.csv")
    
    # Ensure text column contains strings
    dataset['text'] = dataset['text'].astype(str)
    
    # Set maximum sequence length
    max_length = 128
    
    # Tokenize and pad sequences
    print("Tokenizing and padding sequences...")
    encoded_data = tokenize_and_pad(dataset['text'], max_length)
    
    # Convert to numpy arrays
    input_ids = encoded_data['input_ids'].numpy()
    attention_mask = encoded_data['attention_mask'].numpy()
    
    # Split data
    print("Splitting data into train and test sets...")
    X_train_ids, X_test_ids, y_train, y_test = train_test_split(
        input_ids,
        dataset['sentiment'].values,
        test_size=0.2,
        random_state=42
    )
    
    # Split attention mask
    X_train_attention, X_test_attention = train_test_split(
        attention_mask,
        test_size=0.2,
        random_state=42
    )
    
    # Free up memory
    del input_ids
    del attention_mask
    del encoded_data
    
    # Create and compile model
    print("Creating model...")
    model = create_model(max_length, tf_bert_model)
    model.summary()

    # Train model with reduced batch size and early stopping
    print("Training model...")
    epochs = 10
    batch_size = 8  # Reduced batch size to match ELMo setup
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        [X_train_ids, X_train_attention],
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, [X_test_ids, X_test_attention], y_test)

if __name__ == "__main__":
    main()