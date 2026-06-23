# Fake vs Real News Classification with Deep Learning

A deep learning project that classifies news articles as **fake** or **real** using Natural Language Processing (NLP) and a neural network built with TensorFlow/Keras.

## Overview

This project demonstrates a complete NLP pipeline:

1. **Load & explore** the dataset
2. **Preprocess text** — stemming, stopword removal
3. **Extract features** — Bag-of-Words with CountVectorizer
4. **Train a neural network** — feedforward classifier with Keras
5. **Evaluate** — accuracy, precision, recall, F1-score, confusion matrix

## Dataset

The [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle contains ~45,000 articles:

| File | Articles | Description |
|------|----------|-------------|
| `True.csv` | 21,417 | Real news articles (Reuters) |
| `Fake.csv` | 23,481 | Fake news articles |

Each article has: `title`, `text`, `subject`, `date`

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/DL-Fake-And-Real-News.git
cd DL-Fake-And-Real-News
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place the CSV files in a `data/` folder:

```
DL-Fake-And-Real-News/
├── data/
│   ├── True.csv
│   └── Fake.csv
├── fake-and-real-news.ipynb
├── requirements.txt
└── README.md
```

### 4. Run the notebook

```bash
jupyter notebook fake-and-real-news.ipynb
```

## Model Architecture

```
Input (50,000 features) → Dense(128, ReLU) → Dropout(0.3)
                        → Dense(64, ReLU)  → Dropout(0.3)
                        → Dense(32, ReLU)
                        → Dense(1, Sigmoid) → Output (Fake/Real)
```

## Results

| Metric | Score |
|--------|-------|
| Accuracy | ~90% |
| Precision | ~90% |
| Recall | ~90% |
| F1-Score | ~90% |

## Technologies

- Python 3.8+
- TensorFlow / Keras
- scikit-learn
- NLTK
- pandas, NumPy, Matplotlib, Seaborn

## License

This project is for educational purposes.
