# Amazon Video Games – Recommender System & Sentiment Analysis  
Machine Learning • NLP • Recommender Systems • Python

This project analyzes Amazon Video Games review data through a complete ML/NLP pipeline, integrating **collaborative filtering**, **content-based recommendation**, and **sentiment classification**.

The work is structured in three modules:

---

## 1. Collaborative Filtering (Base Project)  
- Preprocessing of the user–item rating matrix  
- KNN-based item-item recommender with cosine similarity  
- Hyperparameter exploration (K = 20–80)  
- Best configuration: **item-based CF, cosine similarity, K=40**, achieving **RMSE = 1.04**  
- User segmentation with K-Means (limited interpretability due to sparsity)  
- Comparison with **Matrix Factorization**, which achieved slightly better predictive performance  

---

## 2. Content-Based Recommendation (Intermediate Project)  
- Textual representation of each product (title + description)  
- Two embedding techniques:
  - **TF-IDF**
  - **Transformers-based embeddings (BERT-like model)**
- Recommendation based on item–item similarity combined with user preferences  
- Evaluation:
  - Overlap TF-IDF vs Transformers: **0.15** → low agreement  
  - Overlap Content-Based vs Collaborative: **≈ 0** → complementary systems  
- Popularity analysis showed collaborative filtering tends to recommend repeated items more frequently, while content-based systems produce more diversified suggestions

---

## 3. Sentiment Analysis (Advanced Project)  
- Reviews labeled as:
  - 1–2 ⭐ → **negative**
  - 3 ⭐ → **neutral**
  - 4–5 ⭐ → **positive**
- Embedding methods:
  - **TF-IDF**
  - **Transformers**
- Classifiers:
  - **Logistic Regression**
  - **Random Forest**
- Results:
  - Accuracy > **80%** across models  
  - Best performance on positive and negative classes  
  - Neutral class harder due to dataset imbalance  
- Macro and weighted **F1-scores** give a more reliable view of model quality

---

## Key Takeaways
- Collaborative filtering excels when sufficient user–item interactions exist  
- Content-based models offer a complementary perspective by leveraging textual information  
- Transformers capture semantic similarity significantly better than TF-IDF  
- The combination of recommender systems + NLP provides a richer understanding of user preferences  
- Data quality, representation, and evaluation metrics are critical for reliable outcomes  

---


## Technologies & Skills
- **Python**
- **Pandas, NumPy**
- **Scikit-Learn**
- **NLP: TF-IDF, Transformers**
- **Recommender Systems: KNN, Cosine Similarity, Matrix Factorization**
- **Sentiment Classification**
- **Evaluation Metrics: RMSE, Precision, Recall, F1**
- **Visualization: Matplotlib, Seaborn**

---

## Materials
- Full report (PDF)
- Python script

---

## Author
Davide Frosi


