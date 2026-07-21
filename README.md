# 🧠 NLP Text Processing Dashboard

A Streamlit web application that demonstrates the fundamentals of **Natural Language Processing (NLP)** by cleaning text data, removing stopwords, and converting text into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

---

## 📌 Project Overview

This project introduces the basic NLP preprocessing pipeline using real product reviews. It allows users to explore the dataset, understand NLP concepts, preprocess text, and visualize the most important words extracted from the reviews.

---

## 🚀 Features

- 📄 Display raw product reviews
- 🧹 Text preprocessing
- 🔠 Convert text to lowercase
- ✂️ Remove punctuation
- 🚫 Remove English stopwords
- 🔢 Generate TF-IDF vectors
- 📊 Display TF-IDF table
- 📈 Visualize the most important words
- 🌐 Interactive Streamlit interface

---

## 📂 Dataset

The project uses a dataset containing customer product reviews.

The application processes the **Text** column to prepare it for Machine Learning and NLP tasks.

---

## 🧠 NLP Techniques Used

### Text Cleaning
- Convert text to lowercase
- Remove punctuation

### Stopword Removal
- Remove common English words such as:
  - the
  - is
  - and
  - in
  - of

### Feature Extraction

- TF-IDF Vectorization

TF-IDF converts textual data into numerical vectors that Machine Learning models can understand.

---

## 📊 Application Sections

### 🧠 What is NLP

Explains:

- Natural Language Processing
- Why NLP is important
- Real-world applications

---

### 📄 General Information

- Number of reviews
- Missing values
- Duplicate removal
- Preview of the dataset

---

### 🧹 NLP Processing

Shows:

- Original Text
- Cleaned Text
- Processed Text after removing stopwords

---

### 🔢 TF-IDF

Displays:

- TF-IDF Matrix
- Top 10 Important Words
- Bar Chart of Word Importance

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NLTK
- Scikit-learn

---

## 📦 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/NLP-Text-Processing-Dashboard.git
cd NLP-Text-Processing-Dashboard
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download the required NLTK stopwords

```python
import nltk
nltk.download("stopwords")
```

Run the application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
NLP-Text-Processing-Dashboard/
│
├── app.py
├── Reviews.csv
├── requirements.txt
├── README.md
└── assets/
```

---

## 🔄 Workflow

1. Load Reviews Dataset
2. Remove Duplicate Records
3. Remove Missing Values
4. Clean Text
5. Remove Stopwords
6. Generate TF-IDF Features
7. Display TF-IDF Matrix
8. Visualize Top Important Words

---

## 🎯 Future Improvements

- Stemming
- Lemmatization
- Word Cloud Visualization
- Sentiment Analysis
- Topic Modeling (LDA)
- Text Classification
- Word Embeddings (Word2Vec, GloVe)
- Transformer Models (BERT)

---

## 👨‍💻 Author

**Mohammad Gharaibah**

Computer Science Student

Interested in:

- Artificial Intelligence
- Natural Language Processing
- Machine Learning
- Data Science

---

## ⭐ Support

If you found this project useful, please give it a ⭐ on GitHub!
```
