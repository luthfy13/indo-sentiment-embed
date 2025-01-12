# Comparative Analysis of BERT and ELMo for Indonesian Sentiment Analysis

This repository contains the implementation of a comparative study between BERT and ELMo embeddings for sentiment analysis of Indonesian text, specifically focusing on e-commerce reviews.

## Overview

The project implements and compares two contextual embedding approaches (BERT and ELMo) using two different classification methods (BiLSTM and Multinomial Naive Bayes) for sentiment analysis of Indonesian text. The implementation includes comprehensive preprocessing, model training, and evaluation components.

## Research Details

- **Dataset**: PRDICT-ID (Indonesian e-commerce reviews from Tokopedia)
- **Models**: 
  - BERT (IndoBERT base-uncased)
  - ELMo (TensorFlow Hub)
- **Classifiers**:
  - Bidirectional LSTM
  - Multinomial Naive Bayes
- **Results published**: The 6th East Indonesia Conference on Computer and Information Technology (EIConCIT) 2024

## Requirements

```
tensorflow>=2.8.0
transformers>=4.18.0
tensorflow-hub>=0.12.0
tensorflow-text>=2.8.0
numpy>=1.21.0
pandas>=1.4.0
scikit-learn>=1.0.0
seaborn>=0.11.0
matplotlib>=3.5.0
tqdm>=4.64.0
```

## Project Structure

```
.
├── main7.py               # Bi-LSTM+ELMo implementation
├── main8.py               # Bi-LSTM+BERT implementation
├── main7a.py              # MNB+ELMo implementation
├── main8b.py              # MNB+BERT implementation
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Implementation Details

### Model Architectures

#### BiLSTM Classifier
- First BiLSTM layer: 32 units (return_sequences=True)
- Second BiLSTM layer: 16 units
- Dense layer: 8 units (ReLU activation)
- Output layer: 1 unit (sigmoid activation)

#### MNB Classifier
- M-estimate smoothing (m=2)
- PCA reduction to 256 dimensions
- L2 normalization
- Min-max scaling

### Performance Results

BiLSTM Classifier Results:
- BERT: 96% accuracy
- ELMo: 84% accuracy

MNB Classifier Results:
- BERT: 78% accuracy
- ELMo: 59% accuracy

## Setup and Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analysis-bert-elmo.git
cd sentiment-analysis-bert-elmo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the implementations:

For ELMo:
```bash
python main7.py
```

For BERT:
```bash
python main8.py
```

## Hardware Requirements

Recommended specifications:
- GPU: NVIDIA RTX 3090TI or similar
- CPU: 3.8GHz, 8 Cores
- RAM: 16GB
- Storage: 500GB

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{ilmawan2024comparative,
  title={Comparative Analysis of BERT and ELMo Embeddings for Indonesian Sentiment Analysis},
  author={Ilmawan, Lutfi Budi and Indra, Dolly},
  booktitle={The 6th East Indonesia Conference on Computer and Information Technology (EIConCIT)},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Lutfi Budi Ilmawan (Universitas Muslim Indonesia)
- Dr. Ir. Dolly Indra (Universitas Muslim Indonesia)

## Acknowledgments

This research was supported by Universitas Muslim Indonesia. Special thanks to the PRDICT-ID dataset creators for making their data available for research.
