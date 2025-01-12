# reduced complexity
# experiment:
# - embedding          : ELMo only
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
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import keras.mixed_precision as mixed_precision

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

def load_elmo():
    """Load pre-trained ELMo model."""
    elmo_model = hub.load("https://tfhub.dev/google/elmo/3")
    return elmo_model

def pad_or_truncate_text(text, max_words=128):
    """Pad or truncate text to fixed number of words."""
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text

def get_elmo_embeddings(texts, elmo_model, batch_size=32, max_length=128):
    """Get ELMo embeddings for the input texts with consistent dimensionality."""
    embeddings = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating ELMo embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # Preprocess texts to have consistent length
        processed_texts = [pad_or_truncate_text(text, max_length) for text in batch_texts]
        
        try:
            # Get ELMo embeddings for the batch
            batch_embeddings = elmo_model.signatures['default'](
                tf.constant(processed_texts)
            )['elmo']
            
            # Ensure consistent dimensionality
            batch_embeddings = tf.keras.preprocessing.sequence.pad_sequences(
                batch_embeddings,
                maxlen=max_length,
                padding='post',
                dtype='float32'
            )
            
            embeddings.append(batch_embeddings)
            
        except Exception as e:
            print(f"Error processing batch {i}-{i+batch_size}: {str(e)}")
            continue
    
    # Concatenate all batches
    try:
        final_embeddings = np.concatenate(embeddings, axis=0)
        return final_embeddings
    except Exception as e:
        print(f"Error concatenating embeddings: {str(e)}")
        return None

def create_model(max_length, embedding_dim=1024):
    """Create the ELMo + BiLSTM model."""
    # Input layer
    input_layer = Input(shape=(max_length, embedding_dim), dtype='float32', name='input_embeddings')
    
    # BiLSTM layers with reduced complexity
    x = Bidirectional(LSTM(32, return_sequences=True))(input_layer)
    # x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(16))(x)
    # x = Dropout(0.2)(x)
    
    # Dense layers
    x = Dense(8, activation='relu')(x)
    # x = Dropout(0.1)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Create and compile model
    model = Model(inputs=input_layer, outputs=output)
    
    # Configure optimizer with mixed precision
    optimizer = Adam(learning_rate=1e-5)  # Reduced learning rate
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
    # Generate predictions
    y_pred = model.predict(X_test, batch_size=8)  # Reduced batch size for prediction
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
    max_length = 128  # Reduced from original
    
    # Load ELMo model
    print("Loading ELMo model...")
    elmo_model = load_elmo()
    
    # Generate ELMo embeddings with smaller batch size
    print("Generating ELMo embeddings...")
    embeddings = get_elmo_embeddings(
        dataset['text'].values,
        elmo_model,
        batch_size=16,  # Reduced batch size
        max_length=max_length
    )
    
    if embeddings is None:
        print("Error generating embeddings. Exiting.")
        return
    
    # Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        dataset['sentiment'].values,
        test_size=0.2,
        random_state=42
    )
    
    # Free up memory
    del embeddings
    
    # Create and compile model
    print("Creating model...")
    model = create_model(max_length)
    model.summary()

    # Train model with reduced batch size and added callbacks
    print("Training model...")
    epochs = 10
    batch_size = 8  # Reduced batch size
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train,
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
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()