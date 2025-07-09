#  Email Spam Classification using Machine Learning

##  Project Overview

This project aims to classify emails as **Spam** or **Not Spam** using a Machine Learning model. By analyzing the content of emails, the model predicts whether an email is genuine or unwanted (spam), helping users automatically filter out unnecessary messages from their inbox.

---

##  Objective

To develop a spam detection system that classifies incoming emails into:
- `Spam (1)`
- `Not Spam (0)`

---

## Dataset Description

The dataset contains two columns:

| Column | Description |
|--------|-------------|
| `text` | The full content of the email |
| `label` | Target variable — 1 for Spam, 0 for Not Spam |

---

## 🛠️ Tools & Technologies Used

- **Programming Language**: Python  
- **Libraries**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
- **ML Algorithm**: Random Forest Classifier
- **Text Vectorization**: TF-IDF Vectorizer or CountVectorizer

---

## ⚙️ Project Workflow

### 1. Data Loading
- Load the dataset using pandas
- Inspect basic structure and balance of the classes

### 2. Data Preprocessing
- Lowercasing all text
- Removing punctuation, special characters, and stopwords
- Tokenization and lemmatization (optional but recommended)
- Text vectorization using **TF-IDF** or **CountVectorizer**

### 3. Splitting the Dataset
- Use `train_test_split()` from sklearn
- Typical ratio: 80% training, 20% testing

### 4. Model Training
- Fit a **Random Forest Classifier** on the training data

### 5. Evaluation
- Metrics used: 
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

### 6. Visualization
- Visualize confusion matrix and class distribution
- Plot feature importances and word distributions

---

## Sample Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv("spam.csv")

# Features and labels
X = df['text']
y = df['label']

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
