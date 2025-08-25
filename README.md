# 📰 Political News Bias Detection & Clustering

## 📌 Overview
This project analyzes **political news bias** using Natural Language Processing (NLP).  
It has two main components:  

1. **Bias Detection (Classification):** Classify political news articles into categories such as *Left / Center / Right*.  
2. **Bias-Aware Clustering:** Group related news articles using semantic embeddings while incorporating political bias awareness.  

The goal is to show how different outlets report the same story with different political leanings.

---

## ⚙️ Tech Stack
- **Python, PyTorch**
- **Hugging Face Transformers (RoBERTa / DeBERTa)** for bias detection  
- **Sentence-Transformers** for semantic embeddings  
- **Scikit-learn** for clustering and evaluation  
- **Pandas, NumPy** for preprocessing  
- **Matplotlib/Seaborn** for visualization  
- **Jupyter/Colab** for experimentation  

---

## 🚀 Features
- ✅ Preprocess articles (clean, normalize, remove URLs)  
- ✅ Train supervised **bias classifier** (Left / Center / Right)  
- ✅ Generate embeddings with **RoBERTa/DeBERTa**  
- ✅ Cluster articles by semantic similarity with **cosine distance**  
- ✅ **Bias-aware clustering**: boost similarity for same-bias articles  
- ✅ Visualize results (confusion matrix, cluster summaries, bias distribution)  

---


