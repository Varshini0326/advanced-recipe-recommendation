# Advanced Recipe Recommendation System
**With Machine and Deep Learning**  
**Personalized User Experience**

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
- [Installation](#installation)
- [Usage](#usage)

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

## Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/Varshini0326/advanced-recipe-recommendation.git

## Usage
1.Load the dataset in Jupyter Notebook.
2.Preprocess the data.
3.Train collaborative filtering and content-based models.
4.Run sentiment analysis on user reviews.
5.Generate hybrid recommendations.
6.Visualize results in Power BI.
