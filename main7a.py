import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.optimizers import Adam
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import keras.mixed_precision as mixed_precision

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

class OptimizedMNBWithMEstimate:
    def __init__(self, m=2):
        self.m = m
        self.class_priors = None
        self.feature_probs = None
        self.classes = None
        self.pca = PCA(n_components=256)  # Reduce dimensions
        
    def preprocess_features(self, X, fit=False):
        # L2 normalization
        X_normalized = normalize(X, norm='l2')
        
        # PCA transformation
        if fit:
            X_reduced = self.pca.fit_transform(X_normalized)
        else:
            X_reduced = self.pca.transform(X_normalized)
            
        # Scale to [0, 1] range
        X_scaled = (X_reduced - X_reduced.min(axis=0)) / (X_reduced.max(axis=0) - X_reduced.min(axis=0) + 1e-10)
        return X_scaled
        
    def fit(self, X, y):
        # Preprocess features
        X_processed = self.preprocess_features(X, fit=True)
        
        self.classes = np.unique(y)
        n_samples, n_features = X_processed.shape
        
        # Calculate class priors with smoothing
        self.class_priors = {}
        class_counts = {}
        for c in self.classes:
            class_counts[c] = np.sum(y == c)
            # Smooth class priors
            self.class_priors[c] = (class_counts[c] + self.m/len(self.classes)) / (n_samples + self.m)
        
        # Calculate feature probabilities with m-estimate smoothing
        self.feature_probs = {}
        for c in self.classes:
            class_samples = X_processed[y == c]
            
            # Sum of feature values for this class
            feature_sums = np.sum(class_samples, axis=0)
            
            # Calculate class-specific prior based on feature distribution
            feature_prior = np.mean(X_processed, axis=0)
            
            # Total sum for normalization
            total_sum = np.sum(feature_sums)
            
            # Calculate smoothed probabilities using m-estimate with class-specific prior
            smoothed_probs = (feature_sums + self.m * feature_prior) / (total_sum + self.m)
            
            # Add small epsilon to avoid zero probabilities
            smoothed_probs = np.clip(smoothed_probs, 1e-10, 1.0)
            
            self.feature_probs[c] = smoothed_probs
            
        return self
    
    def predict_proba(self, X):
        # Preprocess features
        X_processed = self.preprocess_features(X, fit=False)
        
        probas = np.zeros((X_processed.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            # Calculate log probabilities
            log_priors = np.log(self.class_priors[c])
            log_likelihood = np.sum(X_processed * np.log(self.feature_probs[c]), axis=1)
            probas[:, i] = log_priors + log_likelihood
            
        # Convert log probabilities to probabilities
        probas = np.exp(probas - np.max(probas, axis=1)[:, np.newaxis])  # Numerically stable softmax
        probas /= np.sum(probas, axis=1)[:, np.newaxis]
        return probas
    
    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]

def pad_or_truncate_text(text, max_words=128):
    """Pad or truncate text to fixed number of words."""
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text

def get_elmo_embeddings(texts, elmo_model, batch_size=16, max_length=128):
    """Get ELMo embeddings with improved layer combination."""
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating ELMo embeddings"):
        batch_texts = texts[i:i + batch_size]
        processed_texts = [pad_or_truncate_text(text, max_length) for text in batch_texts]
        
        try:
            # Get ELMo embeddings
            elmo_output = elmo_model.signatures['default'](tf.constant(processed_texts))
            
            # Get all three layers (shape: batch_size x seq_length x 3 x 1024)
            batch_embeddings = elmo_output['elmo']
            
            # Combine layers with learned weights
            # Give more weight to upper layers as they capture more semantic information
            layer_weights = [0.2, 0.3, 0.5]  # weights for each layer
            weighted_sum = tf.reduce_sum(batch_embeddings * tf.constant(layer_weights)[None, None, :, None], axis=2)
            
            # Use the last hidden state instead of mean
            last_hidden = weighted_sum[:, -1, :]
            
            # L2 normalize
            normalized_embeddings = tf.nn.l2_normalize(last_hidden, axis=1)
            
            embeddings.append(normalized_embeddings.numpy())
            
        except Exception as e:
            print(f"Error processing batch {i}-{i+batch_size}: {str(e)}")
            continue
    
    try:
        final_embeddings = np.concatenate(embeddings, axis=0)
        return final_embeddings
    except Exception as e:
        print(f"Error concatenating embeddings: {str(e)}")
        return None

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def main():
    # Load data
    print("Loading dataset...")
    dataset = pd.read_csv("../data/PRDICT-ID/vw_dataset.csv")
    
    # Ensure text column contains strings
    dataset['text'] = dataset['text'].astype(str)
    
    # Load ELMo model
    print("Loading ELMo model...")
    elmo_model = hub.load("https://tfhub.dev/google/elmo/3")
    
    # Get ELMo embeddings
    print("Generating ELMo embeddings...")
    embeddings = get_elmo_embeddings(
        dataset['text'].values,
        elmo_model,
        batch_size=16,
        max_length=128
    )
    
    if embeddings is None:
        print("Error generating embeddings. Exiting.")
        return
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        dataset['sentiment'].values,
        test_size=0.2,
        random_state=42,
        stratify=dataset['sentiment'].values  # Ensure balanced split
    )
    
    # Free up memory
    del embeddings
    
    # Train optimized MNB with m-estimate
    print("Training Multinomial Naive Bayes...")
    mnb = OptimizedMNBWithMEstimate(m=2)
    mnb.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = mnb.predict(X_test)
    y_pred_proba = mnb.predict_proba(X_test)[:, 1]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    print("\nConfusion Matrix:")
    plot_confusion_matrix(y_test, y_pred)
    
    # Calculate and print AUC-ROC score
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC-ROC Score: {auc_roc:.4f}")

if __name__ == "__main__":
    main()