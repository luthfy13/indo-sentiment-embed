import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import TFBertModel, BertTokenizer
from tqdm import tqdm
import tensorflow as tf
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

def tokenize_and_encode(texts, tokenizer, max_length=128):
    """Tokenize and encode texts with consistent padding."""
    # Convert numpy array or pandas series to list
    if isinstance(texts, (np.ndarray, pd.Series)):
        texts = texts.tolist()
    
    # Tokenize all texts at once to get consistent padding
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    
    return encoded['input_ids'].numpy(), encoded['attention_mask'].numpy()

def get_bert_embeddings(texts, bert_model, tokenizer, max_length=128, batch_size=8):
    """Get BERT embeddings for texts."""
    input_ids, attention_mask = tokenize_and_encode(texts, tokenizer, max_length)
    
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting BERT embeddings"):
        batch_input_ids = input_ids[i:i + batch_size]
        batch_attention_mask = attention_mask[i:i + batch_size]
        
        # Get BERT embeddings
        outputs = bert_model([batch_input_ids, batch_attention_mask])[0]
        
        # Use CLS token embedding (first token)
        cls_embeddings = outputs[:, 0, :].numpy()
        all_embeddings.append(cls_embeddings)
    
    return np.concatenate(all_embeddings, axis=0)

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
    
    # Load BERT model and tokenizer
    print("Loading BERT model...")
    bert_model_name = "indolem/indobert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = TFBertModel.from_pretrained(bert_model_name, from_pt=True)
    
    # Get BERT embeddings
    print("Generating BERT embeddings...")
    embeddings = get_bert_embeddings(dataset['text'].values, bert_model, tokenizer)
    
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