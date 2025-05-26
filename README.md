# 🎾 Predicting Success in Professional Tennis with Machine Learning

This project applies machine learning techniques to analyze professional men's tennis player data from 2010 to 2023, aiming to identify which attributes—such as height, surface, serve performance, and age—contribute most to success, and ultimately to **predict player ranking**.

**Author:** Sebastian Buxman  
**Institution:** Weber State University  
**Course:** CS 6600 - Final Project

---

## 🧠 Project Goals

1. **Investigate correlations** between surface types and serve effectiveness (aces, double faults).
2. Explore **relationships between height, age, and success metrics**.
3. Analyze the **importance of break points saved** in match outcomes.
4. Use ML algorithms to **predict whether a player ranks inside the ATP top 100**.
5. Identify **feature importance** to understand what contributes most to high rankings.

---

## 📊 Dataset

- Data source: [Jeff Sackmann’s tennis_atp repository](https://github.com/JeffSackmann/tennis_atp)
- Coverage: All ATP, Challenger, and Futures-level tournaments from **2010 to 2023**
- Aggregated statistics by player name and enriched with each player's latest available ranking

---

## 📌 Key Features Used

- **Height**
- **Surface**
- **Aces, Double Faults**
- **Break Points Saved**
- **Match Duration**
- **Age**
- **Total Wins**
- **Total Matches Played**

---

## 🧪 Models & Techniques

- **Random Forest Regressor** – main model for feature importance and prediction
- **SVM Regressor**
- **MLP Regressor (Neural Network)**
- **Decision Tree Regressor**
- **AdaBoost** (with MLP)

> Data preprocessing included standardization with `StandardScaler`, and a train/test split. Dimensionality reduction was not needed due to a moderate dataset size (~12,587 rows).

---

## 🧠 Findings & Insights

### ✅ Confirmed Hypotheses
- **Height** correlates strongly with **number of aces**
- **Surface type** impacts ace count (fewer on clay, more on grass/carpet)
- **Break points saved** is a strong indicator of match outcome, especially in early and late rounds

### ❌ Rejected Hypotheses
- **Age** is *not* a strong predictor of success
- **Double faults** are *not* strongly correlated with height or surface

### 🎯 Feature Importance (from Random Forest)
- `total_wins` was the most important feature
- When removing zeros (inactive or low-activity players), **loser's break points saved (`l_bpSaved`)** became most predictive

### 🔍 Model Performance
- Models performed **best for predicting top 300 players**, where more data was available
- Predicting the entire ATP ranking with high accuracy remains a challenge

---

## 📈 Visualizations Included

- Winner height vs. aces (with surface overlays)
- Surface type vs. aces (boxplots)
- Break points saved vs. match round
- Age difference distribution by round
- Feature importance plots
- Predicted vs. actual ranking (with and without 0s)
