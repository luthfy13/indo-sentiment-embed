# Indonesian Sentiment Analysis: Comparative Analysis of BERT and ELMo

This repository contains the implementation and analysis of a comparative study between BERT (Bidirectional Encoder Representations from Transformers) and ELMo (Embeddings from Language Models) for sentiment analysis of Indonesian text. The study focuses on analyzing e-commerce reviews using the PRDICT-ID dataset from Tokopedia.

## Project Overview

The project implements and compares two state-of-the-art contextual embedding models for sentiment analysis:
- ELMo implementation using TensorFlow Hub
- BERT implementation using IndoBERT (indobert-base-uncased)

Both models are integrated with a BiLSTM classification architecture to perform sentiment analysis on Indonesian language text.

## Dataset

We use the PRDICT-ID dataset, which contains e-commerce reviews from Tokopedia. The dataset was obtained from the Data in Brief journal (Q2-ranked publication) and includes:
- Review text in Indonesian language
- Binary sentiment labels (positive/negative)
- Original dataset size: 5,400 reviews

## Technical Implementation

### Common Configuration

Both implementations share the following configuration:

```python
# Model Architecture
- Maximum sequence length: 128 tokens
- BiLSTM layers: 32 units → 16 units
- Dense layers: 8 units → 1 unit (sigmoid)

# Training Parameters
- Optimizer: Adam (lr=1e-5)
- Loss: Binary crossentropy
- Batch size: 8
- Epochs: 10
- Validation split: 0.2
```

### ELMo Implementation (main7.py)

Key features:
- Pre-trained ELMo model from TensorFlow Hub
- Batch processing for embedding generation
- Custom text padding and truncation
- Memory optimization through batch processing
- 281,233 trainable parameters

### BERT Implementation (main8.py)

Key features:
- IndoBERT base model (uncased)
- BertTokenizer for text processing
- Dual input handling (input_ids and attention_mask)
- PyTorch to TensorFlow model conversion
- 110,773,905 trainable parameters

## Performance Results

### Model Comparison

| Metric             | ELMo   | BERT   | Difference |
|-------------------|--------|--------|------------|
| Training Accuracy | 86.11% | 99.62% | +13.51%    |
| Validation Accuracy| 83.56% | 97.69% | +14.13%    |
| Test Accuracy     | 84%    | 96%    | +12%       |
| False Positives   | 101    | 8      | -93        |
| Training Time/Epoch| 18s    | 65s    | +47s       |

### Key Findings

1. Performance:
   - BERT consistently outperformed ELMo across all metrics
   - Most significant improvement in false positive reduction
   - Higher stability in learning patterns

2. Resource Requirements:
   - BERT requires significantly more computational resources
   - BERT has ~394x more parameters than ELMo
   - ELMo shows faster training and inference times

## Project Structure

```
├── main7.py                 # ELMo implementation
├── main8.py                 # BERT implementation
├── requirements.txt         # Project dependencies
└── data/
    └── PRDICT-ID/          # Dataset directory
        └── vw_dataset.csv  # Preprocessed dataset
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Transformers
- TensorFlow Hub
- NumPy
- Pandas
- Scikit-learn
- CUDA-capable GPU (recommended)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/indo-sentiment-analysis.git

# Install dependencies
pip install -r requirements.txt
```

## Usage

To run the ELMo implementation:
```bash
python main7.py
```

To run the BERT implementation:
```bash
python main8.py
```

## Results Visualization

Both implementations include visualization tools for:
- Training/validation accuracy curves
- Training/validation loss curves
- Confusion matrices
- Classification reports

## Conclusion

This study demonstrates the trade-offs between model complexity and performance in Indonesian sentiment analysis. While BERT shows superior performance across all metrics, ELMo provides a viable alternative for resource-constrained environments or applications requiring faster processing times.

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{ilmawan2024sentiment,
  title={Analysis of BERT and ELMo Performance in Indonesian Sentiment Classification},
  author={Ilmawan, Lutfi Budi and Indra, Dolly},
  journal={Indonesian Journal of Computing and Cybernetics Systems},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
