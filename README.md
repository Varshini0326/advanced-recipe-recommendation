# Advanced Recipe Recommendation System
**With Machine and Deep Learning - Personalized User Experience**

A hybrid recipe recommendation system that delivers personalized recipes based on preferences, dietary needs, and review sentiment using machine learning and deep learning.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub Repo Size](https://img.shields.io/github/repo-size/Varshini0326/advanced-recipe-recommendation)

## Table of Contents
- [Abstract](#abstract)
- [Keywords](#keywords)
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Author](#author)
  
## Abstract
This project develops an advanced recipe recommendation system that provides personalized suggestions to enhance user experience. By combining **sentiment analysis**, **collaborative filtering**, and **content-based filtering**, the system delivers relevant recipes based on user preferences, dietary restrictions, and past interactions. Integration of textual reviews and numerical ratings ensures accurate, engaging, and customized recommendations.

## Keywords
Recipe Recommendation System, Sentiment Analysis, Feature Engineering, User Reviews, Recipe Ratings, Collaborative Filtering, Content-Based Filtering, Natural Language Processing, Deep Learning, Machine Learning, Personalization, User Experience, Dietary Needs, API Integration, Power BI

## Overview
Hybrid Recipe Recommendation System providing personalized recipe suggestions based on user preferences, dietary needs, and review sentiment, leveraging machine learning and deep learning for enhanced user experience.

## Features
- Personalized recipe recommendations
- Collaborative filtering using SVD
- Content-based filtering using TF-IDF and cosine similarity
- Sentiment analysis using RNN/LSTM
- Hybrid model combining multiple recommendation techniques

## Technologies Used
- Python (pandas, NumPy, scikit-learn, TensorFlow/Keras)
- Power BI for visualization
- Jupyter Notebooks for experimentation
- Git and GitHub for version control
  
## System Architecture
The system consists of the following modules:
1. **Data Collection & Preprocessing** – Collects recipe data, user ratings, and reviews; cleans and transforms the data.
2. **Collaborative Filtering** – Suggests recipes based on user-item interactions using SVD.
3. **Content-Based Filtering** – Suggests recipes based on similarity of recipe features using TF-IDF and cosine similarity.
4. **Sentiment Analysis** – Analyzes textual reviews using RNN/LSTM to capture user preferences.
5. **Hybrid Recommendation Model** – Combines outputs from all models to provide final recommendations.
6. **Visualization & Reporting** – Uses Power BI for insights and interactive dashboards.

![System Architecture](https://github.com/Varshini0326/advanced-recipe-recommendation/blob/main/Image.png)

# Dataset

This project uses the Recipe Reviews and User Feedback Dataset from the UCI Machine Learning Repository.
🔗 View Dataset on UCI

**Dataset Summary**

Instances: 18,182 reviews

Features: 15 attributes, including recipe ID, user ID, user reputation, review text, up-votes, down-votes, star rating, etc.

Missing Values: Present (some fields contain missing entries, encoded as “2” in the dataset)

License: Creative Commons Attribution 4.0 (CC BY 4.0)

# Load Dataset in Python #
from ucimlrepo import fetch_ucirepo
## Fetch the UCI dataset (ID = 911)##
recipe_reviews = fetch_ucirepo(id=911)
## Features (data) and targets (if any)##
X = recipe_reviews.data.features
y = recipe_reviews.data.targets
## Metadata and variable info ##
print(recipe_reviews.metadata)
print(recipe_reviews.variables)

# Citation

Ali, A., Matuszewski, S., & Czupyt, J. (2023). Recipe Reviews and User Feedback [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5FG95

# Installation

## 1.Clone the repository
git clone https://github.com/Varshini0326/advanced-recipe-recommendation.git
cd advanced-recipe-recommendation
## 2.Install dependencies
pip install -r requirements.txt

## Usage
1. Load the dataset in Jupyter Notebook.
2. Preprocess the data.
3. Train collaborative filtering and content-based models.
4. Run sentiment analysis on user reviews.
5. Generate hybrid recommendations.
6. Visualize results in Power BI.

## License
This project is licensed under the [MIT License](LICENSE).

## Author

👩‍💻 **Varshini Konduru**

- [GitHub](https://github.com/Varshini0326)
- [LinkedIn](https://www.linkedin.com/in/varshini-konduru-310767195/)
