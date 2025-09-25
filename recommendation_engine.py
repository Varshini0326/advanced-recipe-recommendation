# **Loding Data**
"""

import pandas as pd

# Load the CSV file into a DataFrame
data = pd.read_csv('/content/drive/MyDrive/Recipe Reviews and User Feedback Dataset.csv')
data

import pandas as pd

# Load the data
data = pd.read_csv("/content/drive/MyDrive/Recipe Reviews and User Feedback Dataset.csv")

# Rename columns
new_names = {
    'Unnamed: 0': 'recipe_id',  # unique identifier for each recipe
    'recipe_number': 'recipe_number',
    'recipe_code': 'recipe_code',
    'recipe_name': 'recipe_name',
    'comment_id': 'review_id',
    'user_id': 'user_id',
    'user_name': 'username',
    'user_reputation': 'user_reputation',
    'created_at': 'created_at',
    'reply_count': 'reply_count',
    'thumbs_up': 'thumbs_up',
    'thumbs_down': 'thumbs_down',
    'stars': 'star_rating',
    'best_score': 'best_score',
    'text': 'review_text'
}

# Renamed the columns
data = data.rename(columns=new_names)
# Save the data with the new column names
data.to_csv('data_renamed.csv', index=False)

# Print the updated data
data

import pandas as pd

# Initialize the meta_data DataFrame with additional columns for Min and Max values
meta_data = pd.DataFrame(columns=['Description', 'Data Type', 'Unique Values', 'Count', 'Min', 'Max', 'Mean', 'StdDeviation'])

# Dictionary for desc
Description = {
    'recipe_id': 'Index',
    'recipe_number': 'Recipe Number',
    'recipe_code': 'Recipe Code',
    'recipe_name': 'Name of the Recipe',
    'review_id': 'Comment ID',
    'user_id': 'User ID',
    'username': 'User Name',
    'user_reputation': 'User Reputation',
    'created_at': 'Timestamp of Comment Creation',
    'reply_count': 'Number of Replies',
    'thumbs_up': 'Number of Thumbs Up',
    'thumbs_down': 'Number of Thumbs Down',
    'star_rating': 'Star Rating',
    'best_score': 'Best Score',
    'review_text': 'Comment Text'
}

# Populate the metadata DataFrame
meta_data['Data Type'] = data.dtypes
meta_data['Unique Values'] = data.nunique()
meta_data['Count'] = data.count()

for column in data.columns:
    meta_data.loc[column, 'Description'] = Description.get(column, '')

    if pd.api.types.is_numeric_dtype(data[column]):
        meta_data.loc[column, 'Min'] = data[column].min()
        meta_data.loc[column, 'Max'] = data[column].max()
        meta_data.loc[column, 'Mean'] = data[column].mean()
        meta_data.loc[column, 'StdDeviation'] = data[column].std()

# Reset the index to add the column names as a column in the DataFrame
meta_data.reset_index(inplace=True)
meta_data.rename(columns={'index': 'Attribute'}, inplace=True)

# Print the metadata DataFrame
meta_data

data['review_text'].isnull().sum()

import pandas as pd

# Load the data
#data = pd.read_csv("/content/drive/MyDrive/Recipe Reviews and User Feedback Dataset.csv")

# Identify nominal attributes (object type columns)
nominal_attributes = data.select_dtypes(include=['object']).columns

# Calculate the mode for each nominal attribute
mode_values = data[nominal_attributes].mode().iloc[0]

# Set index name and series name for better readability
mode_values.index.name = 'Attribute'
mode_values.name = 'Mode'

# Print the mode values
print("Mode for each nominal attribute:")
mode_values

"""# **Data** **Cleaning**"""

# Checking for missing values in 'text' column
missing_review_text_count = data['review_text'].isnull().sum()
print(f"Number of missing values in 'review_text' column: {missing_review_text_count}")

# Find, display rows with missing 'text' values
if missing_review_text_count > 0:
    missing_review_text_rows = data[data['review_text'].isnull()]
    print("Rows with missing 'review_text' values:")
    print(missing_review_text_rows)

    missing_review_text_indexes = missing_review_text_rows.index
    print("Indexes of rows with missing 'review_text' values:")
    print(missing_review_text_indexes)

#Handle missing values:
data = data.dropna()
data

# Drop rows with missing 'text' values
data = data.dropna(subset=['review_text'])

# Verify that rows with missing 'text' values have been removed
missing_review_text_count_after = data['review_text'].isnull().sum()
print(f"Number of missing values in 'review_text' column after removal: {missing_review_text_count_after}")
data

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# data = pd.read_csv('/content/drive/MyDrive/Recipe Reviews and User Feedback Dataset.csv')

# Display the first few rows of the dataset
data.head()

# Convert created_at to datetime objects
data['created_at'] = pd.to_datetime(data['created_at'], unit='s')

# Rating Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='star_rating', data=data, palette="viridis")
plt.title('Rating Distribution')
plt.xlabel('Star Rating')
plt.ylabel('Frequency')
plt.show()

# User Engagement Analysis (Replies and Thumbs Up)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(data=data, x='reply_count', y='star_rating', hue='thumbs_up', palette="coolwarm")
plt.title('User Engagement: Reply Count vs Star Rating')
plt.xlabel('Reply Count')
plt.ylabel('Star Rating')

plt.subplot(1, 2, 2)
sns.scatterplot(data=data, x='thumbs_up', y='star_rating', hue='thumbs_down', palette="coolwarm")
plt.title('User Engagement: Thumbs Up vs Star Rating')
plt.xlabel('Thumbs Up Count')
plt.ylabel('Star Rating')

plt.tight_layout()
plt.show()

# Temporal Patterns: Analysis of Review Timestamps

# Extract year and month for temporal analysis
data['year'] = data['created_at'].dt.year
data['month'] = data['created_at'].dt.month

# Plotting the number of reviews per year
plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=data, palette="magma")
plt.title('Number of Reviews Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.show()

# Plotting the number of reviews per month (across all years)
plt.figure(figsize=(12, 6))
sns.countplot(x='month', data=data, palette="plasma")
plt.title('Number of Reviews Per Month (Across All Years)')
plt.xlabel('Month')
plt.ylabel('Number of Reviews')
plt.show()

# Analysis of seasonal trends
data['month_year'] = data['created_at'].dt.to_period('M')

plt.figure(figsize=(14, 8))
data.groupby('month_year').size().plot(kind='line', color='green')
plt.title('Temporal Patterns: Review Activity Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.show()

# Correlation Matrix
corr = data[['reply_count', 'thumbs_up', 'thumbs_down', 'star_rating']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
#data = pd.read_csv('/content/drive/MyDrive/Recipe Reviews and User Feedback Dataset.csv')

# Plotting scatter plots for positively correlated pairs
plt.figure(figsize=(18, 6))

# Scatter plot reply_count vs thumbs_up
plt.subplot(1, 3, 1)
sns.scatterplot(x='reply_count', y='thumbs_up', data=data)
plt.title('reply_count vs. thumbs_up')

# Scatter plot reply_count vs thumbs_down
plt.subplot(1, 3, 2)
sns.scatterplot(x='reply_count', y='thumbs_down', data=data)
plt.title('reply_count vs. thumbs_down')

# Scatter plot thumbs_up vs thumbs_down
plt.subplot(1, 3, 3)
sns.scatterplot(x='thumbs_up', y='thumbs_down', data=data)
plt.title('thumbs_up vs. thumbs_down')

plt.tight_layout()
plt.show()

sns.pairplot(data[['reply_count', 'thumbs_up', 'thumbs_down', 'star_rating']])
plt.show()

#Convert created_at to datetime objects
data['created_at'] = pd.to_datetime(data['created_at'], unit='s')
data



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down','star_rating',	'best_score']] = scaler.fit_transform(data[['user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down','star_rating',	'best_score']])
data

import pandas as pd

# Load the data
#data = pd.read_csv("/content/drive/MyDrive/Recipe Reviews and User Feedback Dataset.csv")

unique_values = data['review_text'].unique()

unique_values_count = len(unique_values)

print(f"Number of unique values in 'review_text' column: {unique_values_count}")

import pandas as pd

# Load the data
#data = pd.read_csv("/content/drive/MyDrive/Recipe Reviews and User Feedback Dataset.csv")

unique_values = data['review_text'].unique()
unique_values

# Basic statistics
print(data.describe())

# Plot distribution of star ratings
import matplotlib.pyplot as plt
data['star_rating'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Star Rating')
plt.ylabel('Count')
plt.title('Distribution of Star Ratings')
plt.show()

data['review_text'] = data['review_text'].fillna('').astype(str)
data

for attribute in data.columns:
 missing_values = data[attribute].isnull().sum()
print(f"Attribute: {attribute}")
print(f"Missing values: {missing_values}")

import pandas as pd

#data = pd.read_csv("/content/data_renamed.csv")

for attribute in data.columns:
  missing_values_count = data[attribute].isnull().sum()
  print(f"Attribute: {attribute}, Missing values: {missing_values_count}")

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the data
# data = pd.read_csv("/content/drive/MyDrive/Recipe Reviews and User Feedback Dataset.csv")

# Most liked recipes
most_liked_recipes = data.groupby('recipe_name')['thumbs_up'].sum().sort_values(ascending=False).head(10)
print("Most Liked Recipes:\n", most_liked_recipes)

# Frequently viewed recipes (based on the number of reviews)
frequently_viewed_recipes = data['recipe_name'].value_counts().head(10)
print("Frequently Viewed Recipes:\n", frequently_viewed_recipes)

# Define new custom colors
custom_color_liked = 'green'
custom_color_viewed = 'blue'

# Plot most liked recipes
plt.figure(figsize=(10, 6))
most_liked_recipes.plot(kind='bar', color=custom_color_liked)
plt.title('Top 10 Most Liked Recipes')
plt.xlabel('Recipe Name')
plt.ylabel('Total Thumbs Up')
plt.xticks(rotation=45)
plt.show()

# Plot frequently viewed recipes
plt.figure(figsize=(10, 6))
frequently_viewed_recipes.plot(kind='bar', color=custom_color_viewed)
plt.title('Top 10 Frequently Viewed Recipes')
plt.xlabel('Recipe Name')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the data
#data = pd.read_csv("/content/drive/MyDrive/Recipe Reviews and User Feedback Dataset.csv")

# Most liked recipes
most_liked_recipes = data.groupby('recipe_name')['thumbs_up'].sum().sort_values(ascending=False).head(10)
print("Most Liked Recipes:\n", most_liked_recipes)

# Frequently viewed recipes (based on the number of reviews)
frequently_viewed_recipes = data['recipe_name'].value_counts().head(10)
print("Frequently Viewed Recipes:\n", frequently_viewed_recipes)

# Plot most liked recipes
plt.figure(figsize=(10, 6))
most_liked_recipes.plot(kind='bar')
plt.title('Top 10 Most Liked Recipes')
plt.xlabel('Recipe Name')
plt.ylabel('Total Thumbs Up')
plt.xticks(rotation=45)
plt.show()

# Plot frequently viewed recipes
plt.figure(figsize=(10, 6))
frequently_viewed_recipes.plot(kind='bar')
plt.title('Top 10 Frequently Viewed Recipes')
plt.xlabel('Recipe Name')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.show()

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to a single string
    return ' '.join(tokens)

# Ensure all review_text entries are strings
data.loc[:, 'review_text'] = data['review_text'].astype(str)

# Apply the preprocessing to the review_text column
data.loc[:, 'processed_review_text'] = data['review_text'].apply(preprocess_text)

# vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['processed_review_text'])
data

for attribute in data.columns:
 missing_values = data[attribute].isnull().sum()
print(f"Attribute: {attribute}")
print(f"Missing values: {missing_values}")

# standardized
from sklearn.preprocessing import StandardScaler

# Select numerical features to standardize
numerical_features = data[['user_reputation', 'thumbs_up', 'thumbs_down', 'star_rating', 'best_score']]

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the numerical features
scaled_features = scaler.fit_transform(numerical_features)

# Combine the scaled numerical features with the vectorized text features
import numpy as np
X_combined = np.hstack((X.toarray(), scaled_features))

# Now X_combined can be used for sentiment analysis
scaled_features_df = pd.DataFrame(scaled_features, columns=numerical_features.columns)
scaled_features_df

"""# **Feature Engineering**

# **Sentiment Analysis**
"""

from textblob import TextBlob

# Function to get sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.0:
        return 'Positive'
    elif analysis.sentiment.polarity ==0:
        return 'Neutral'
    else:
        return 'Negative'

# Apply sentiment analysis
data['sentiment'] = data['processed_review_text'].dropna().apply(get_sentiment)
data
# Display sentiment counts
sentiment_counts = data['sentiment'].value_counts()
print("Sentiment Counts:\n", sentiment_counts)

# Plot sentiment counts
plt.figure(figsize=(6, 6))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Sentiment Distribution of Reviews')
plt.ylabel('')
plt.show()

data



# Convert 'created_at' to datetime
data['created_at'] = pd.to_datetime(data['created_at'], unit='s')

# Group by date and calculate the mean sentiment polarity
data['sentiment_polarity'] = data['processed_review_text'].dropna().apply(lambda x: TextBlob(x).sentiment.polarity)
sentiment_over_time = data.groupby(data['created_at'].dt.to_period('M'))['sentiment_polarity'].mean()

# Plot sentiment over time
plt.figure(figsize=(12, 6))
sentiment_over_time.plot(kind='line')
plt.title('Average Sentiment Polarity Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Polarity')
#plt.grid(True)
plt.show()

!pip install vaderSentiment textblob transformers

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Applying VADER sentiment analysis to the 'text' column
data['sentiment_score'] = data['processed_review_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if pd.notna(x) else None)

# DataFrame with the new 'sentiment_score' column
data

data.to_csv("data_with_sentiment.csv", index=False)

print(data.columns)

from google.colab import files

# Save the DataFrame to a CSV file
data.to_csv("data_with_sentiment_1.csv", index=False)

# Download the file
files.download("data_with_sentiment_1.csv")

"""# **Build and Train Models**

# **Content-Based Filtering**
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
#data = pd.read_csv("/content/data_with_sentiment.csv")

# Ensuring sentiment columns are in the DataFrame
if 'sentiment_score' not in data.columns or 'sentiment_polarity' not in data.columns:
    raise ValueError("Sentiment columns are missing from the DataFrame")

# List of features to be used for similarity
features = ['sentiment_score', 'sentiment_polarity', 'star_rating', 'thumbs_up', 'thumbs_down', 'user_reputation']

# Checking if all features are present in the DataFrame
missing_features = [feature for feature in features if feature not in data.columns]
if missing_features:
    print(f"Missing features: {missing_features}")
else:
    # Fill NaN values with 0 or a suitable value using .loc to avoid SettingWithCopyWarning
    data.loc[:, features] = data.loc[:, features].fillna(0)

    # Feature matrix
    feature_matrix = data[features].values

    # Compute cosine similarity
    cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

    # Function to get recommendations based on cosine similarity
    def get_recommendations(recipe_index, cosine_sim, data, top_n=15):
        sim_scores = list(enumerate(cosine_sim[recipe_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]  # Exclude the first one (itself)

        recipe_indices = [i[0] for i in sim_scores]
        return data.iloc[recipe_indices]

    recipe_index = 5 # Index of the recipe for which we want recommendations
    recommendations = get_recommendations(recipe_index, cosine_sim, data)

    recommendations.head(14299)
    print(f"Recommended recipes for recipe at index {recipe_index}:\n", recommendations)

data

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
#data = pd.read_csv("/content/data_with_sentiment.csv")

# Ensuring the necessary columns are in the DataFrame
required_columns = [
    'sentiment_score', 'sentiment_polarity', 'star_rating',
    'thumbs_up', 'thumbs_down', 'user_reputation'
]
if not all(column in data.columns for column in required_columns):
    raise ValueError(f"One or more required columns are missing from the DataFrame: {required_columns}")

# Fill NaN values with 0
data[required_columns] = data[required_columns].fillna(0)

# Creating the feature matrix using the required columns
feature_matrix = data[required_columns].values

# Calculating cosine similarity between the recipes
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# Function to get recommendations based on cosine similarity
def get_recommendations(recipe_index, cosine_sim, data, top_n=10):
    sim_scores = list(enumerate(cosine_sim[recipe_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the first one (itself)

    recipe_indices = [i[0] for i in sim_scores]
    return data.iloc[recipe_indices]

# Function to get recommendations by recipe ID
def get_recommendations_by_id(recipe_id, cosine_sim, data, top_n=20):
    if recipe_id not in data['recipe_number'].values:
        raise ValueError(f"Recipe ID {recipe_id} not found in the dataset.")

    idx = data[data['recipe_number'] == recipe_id].index[0]
    return get_recommendations(idx, cosine_sim, data, top_n)


recipe_id = 5  # recipe ID
recommendations_by_id = get_recommendations_by_id(recipe_id, cosine_sim, data)

# Display the recommended recipes
if not recommendations_by_id.empty:
    print(f"Recommended recipes for recipe ID {recipe_id}:\n", recommendations_by_id[['recipe_number', 'recipe_name', 'sentiment_score','sentiment', 'star_rating']])
else:
    print("No recommendations available due to missing recipe ID.")

# Remove duplicate recipe names
recommendations_by_id = recommendations_by_id.drop_duplicates(subset='recipe_name')

print(f"Recommended recipes for recipe ID {recipe_id}:\n", recommendations_by_id[['recipe_number', 'recipe_name','processed_review_text', 'sentiment_score', 'sentiment','star_rating']])
# Save recommendations to CSV for Power BI
#recommendations_by_id[['recipe_number', 'recipe_name', 'sentiment_score', 'star_rating']].to_csv('recommendations_cosine.similarity.csv', index=False)

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
#data = pd.read_csv("/content/data_with_sentiment.csv")

# Ensure the necessary columns are in the DataFrame
required_columns = [
    'sentiment_score', 'sentiment_polarity', 'star_rating',
    'thumbs_up', 'thumbs_down', 'user_reputation'
]
if not all(column in data.columns for column in required_columns):
    raise ValueError(f"One or more required columns are missing from the DataFrame: {required_columns}")

# Fill NaN values with 0
data[required_columns] = data[required_columns].fillna(0)

# Create the feature matrix using the required columns
feature_matrix = data[required_columns].values

# Calculate cosine similarity between the recipes
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# Function to get recommendations based on cosine similarity
def get_recommendations(recipe_index, cosine_sim, data, top_n=10):
    sim_scores = list(enumerate(cosine_sim[recipe_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the first one (itself)

    recipe_indices = [i[0] for i in sim_scores]
    return data.iloc[recipe_indices]

# Function to get recommendations by recipe ID
def get_recommendations_by_id(recipe_id, cosine_sim, data, top_n=15):
    if recipe_id not in data['recipe_number'].values:
        raise ValueError(f"Recipe ID {recipe_id} not found in the dataset.")

    idx = data[data['recipe_number'] == recipe_id].index[0]
    return get_recommendations(idx, cosine_sim, data, top_n)


recipe_id = 20  # recipe ID
recommendations_by_id = get_recommendations_by_id(recipe_id, cosine_sim, data)

# Remove duplicate recipe names
recommendations_by_id = recommendations_by_id.drop_duplicates(subset='recipe_name')

# Display the recommended recipes
if not recommendations_by_id.empty:
    print(f"Recommended recipes for recipe ID {recipe_id}:\n", recommendations_by_id[['recipe_number', 'recipe_name', 'sentiment_score', 'sentiment', 'star_rating']])
else:
    print("No recommendations available due to missing recipe ID.")

# Save recommendations to CSV for Power BI
#recommendations_by_id[['recipe_number', 'recipe_name', 'sentiment_score', 'star_rating']].to_csv('recommendations_cosine_similarity.csv', index=False)

print(data['recipe_number'].unique())

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load your data
# data = pd.read_csv("data_with_sentiment.csv")

# Ensure the required columns are in the DataFrame
required_columns = ['sentiment_score', 'sentiment_polarity', 'star_rating', 'thumbs_up', 'thumbs_down', 'user_reputation']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    raise ValueError(f"The following required columns are missing from the DataFrame: {missing_columns}")

# Fill NaN values with 0
data.loc[:, required_columns] = data.loc[:, required_columns].fillna(0)

# Create the feature matrix
feature_matrix = data[required_columns].values

# Compute cosine similarity
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# Function to get recommendations based on cosine similarity
def get_recommendations(recipe_index, cosine_sim, data, top_n=20):
    sim_scores = list(enumerate(cosine_sim[recipe_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the first one (itself)

    recipe_indices = [i[0] for i in sim_scores]
    return data.iloc[recipe_indices]

recipe_index = 0  # Give index in DataFrame
recommendations = get_recommendations(recipe_index, cosine_sim, data)

# Remove duplicate recipe names
recommendations = recommendations.drop_duplicates(subset='recipe_name')

# Save recommendations to CSV for Power BI
#recommendations[['recipe_number', 'recipe_name', 'sentiment_score', 'star_rating']].to_csv('recommendations_cosine_similarity1.csv', index=False)

# Display recommended recipes
print(recommendations[['recipe_number', 'recipe_name', 'processed_review_text', 'sentiment', 'sentiment_score', 'sentiment_polarity']])

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load your data
#data = pd.read_csv("data_with_sentiment.csv")

# Ensure the required columns are in the DataFrame
required_columns = ['sentiment_score', 'sentiment_polarity', 'star_rating', 'thumbs_up', 'thumbs_down', 'user_reputation']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    raise ValueError(f"The following required columns are missing from the DataFrame: {missing_columns}")

# Fill NaN values with 0
data.loc[:, required_columns] = data.loc[:, required_columns].fillna(0)

# Create the feature matrix
feature_matrix = data[required_columns].values

# Compute cosine similarity
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# Function to get recommendations based on cosine similarity
def get_recommendations(recipe_index, cosine_sim, data, top_n=5):
    sim_scores = list(enumerate(cosine_sim[recipe_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the first one (itself)

    recipe_indices = [i[0] for i in sim_scores]
    return data.iloc[recipe_indices]

#
recipe_index = 0  # Ensure this index is valid in your DataFrame
recommendations = get_recommendations(recipe_index, cosine_sim, data)

# Remove duplicate recipe names
recommendations_by_id = recommendations_by_id.drop_duplicates(subset='recipe_name')

# Save recommendations to CSV for Power BI
#recommendations_by_id[['recipe_number', 'recipe_name', 'sentiment_score', 'star_rating']].to_csv('recommendations_cosine.similarity2.csv', index=False)

# Display recommended recipes
print(recommendations[['recipe_id', 'recipe_name', 'processed_review_text', 'sentiment', 'sentiment_score', 'sentiment_polarity']])
recommendations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

# Load your data
#data = pd.read_csv("data_with_sentiment.csv")

# Ensure the required columns are in the DataFrame
required_columns = ['sentiment_score', 'sentiment_polarity', 'star_rating', 'thumbs_up', 'thumbs_down', 'user_reputation']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    raise ValueError(f"The following required columns are missing from the DataFrame: {missing_columns}")

# Fill NaN values with 0
data.loc[:, required_columns] = data.loc[:, required_columns].fillna(0)

# Create the feature matrix
feature_matrix = data[required_columns].values

# Compute cosine similarity
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# Function to get recommendations based on cosine similarity
def get_recommendations(recipe_index, cosine_sim, data, top_n=50):
    sim_scores = list(enumerate(cosine_sim[recipe_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the first one (itself)

    recipe_indices = [i[0] for i in sim_scores]
    return data.iloc[recipe_indices]

# Function to calculate metrics
def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    f1 = f1_score(true_labels, predicted_labels, average='binary')
    return precision, recall, f1

# Example interactive session:
recipe_index = int(input("Enter a recipe index to get recommendations: "))

# Get recommendations for the selected recipe
recommendations = get_recommendations(recipe_index, cosine_sim, data)

# Simulate or gather true labels and predicted labels
# True labels: Assume a binary relevance (1 for relevant, 0 for not relevant)
true_labels = [1 if row['sentiment_score'] > 0 else 0 for index, row in recommendations.iterrows()]
# Predicted labels based on a simple threshold on sentiment_score
predicted_labels = [1 if score > 0 else 0 for score in recommendations['sentiment_score']]

# Calculate and display metrics
precision, recall, f1 = calculate_metrics(true_labels, predicted_labels)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# Remove duplicate recipe names and save the recommendations to a CSV
recommendations = recommendations.drop_duplicates(subset='recipe_name')

# Save recommendations to CSV for Power BI
recommendations[['recipe_id', 'recipe_name', 'star_rating','month_year','processed_review_text','sentiment','sentiment_score','sentiment_polarity']].to_csv('top 50 recommendations_cosine.similarity.csv', index=False)

# Display recommended recipes
recommendations[['recipe_id', 'recipe_name','star_rating', 'processed_review_text', 'sentiment', 'sentiment_score', 'sentiment_polarity']]

data

"""**2.Collaborative Filtering using Singular Value Decomposition (SVD)**"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Load your data
#data = pd.read_csv('path/to/your/dataset.csv')

# Create user-item matrix
user_item_matrix = data.pivot_table(index='user_id', columns='recipe_id', values='star_rating', fill_value=0)

# Perform SVD
svd = TruncatedSVD(n_components=20)
user_item_matrix_reduced = svd.fit_transform(user_item_matrix)
user_item_matrix_approx = np.dot(user_item_matrix_reduced, svd.components_)

# Convert back to DataFrame
user_item_matrix_approx_df = pd.DataFrame(user_item_matrix_approx, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Function to recommend items
def recommend_items(user_id, user_item_matrix_approx_df, top_n=10):
    if user_id not in user_item_matrix_approx_df.index:
        raise ValueError(f"User ID {user_id} not found in the matrix.")

    user_ratings = user_item_matrix_approx_df.loc[user_id]
    recommendations = user_ratings.sort_values(ascending=False).head(top_n)

    return recommendations

# Example usage
user_id = "u_Lu6p25tmE77j"  # Example user ID
recommended_items = recommend_items(user_id, user_item_matrix_approx_df)
print(f"Recommended items for user {user_id}:\n", recommended_items)

# Check the index of the user-item matrix
print(user_item_matrix.index.tolist())

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Sample data loading (adjust the path and format as needed)
#data = pd.read_csv("/content/data_with_sentiment.csv")

# Ensure data contains necessary columns
required_columns = ['recipe_id', 'star_rating']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Required columns are missing from the dataset.")

# Create the item-user matrix
item_user_matrix = data.pivot_table(index='recipe_id', columns='user_id', values='star_rating')

# Fill NaN values with 0 for SVD processing
item_user_matrix = item_user_matrix.fillna(0)

# Convert the matrix to a sparse matrix format
item_user_matrix_sparse = csr_matrix(item_user_matrix.values)
# Define the number of latent factors
n_factors = 20

# Apply SVD
svd = TruncatedSVD(n_components=n_factors)
item_user_matrix_svd = svd.fit_transform(item_user_matrix_sparse)

# Compute the approximate item-user matrix
item_user_matrix_approx = item_user_matrix_svd.dot(svd.components_)
import numpy as np

# Convert the approximate matrix back to a DataFrame
item_user_matrix_approx_df = pd.DataFrame(item_user_matrix_approx,
                                          index=item_user_matrix.index,
                                          columns=item_user_matrix.columns)

# Function to get item recommendations
def recommend_items(item_id, item_user_matrix_approx_df, top_n=10):
    if item_id not in item_user_matrix_approx_df.index:
        raise ValueError(f"Item ID {item_id} not found in the matrix.")

    # Predict ratings for the given item
    item_ratings = item_user_matrix_approx_df.loc[item_id]

    # Get the top_n items with the highest predicted ratings
    recommendations = item_ratings.sort_values(ascending=False).head(top_n)

    return recommendations

# Example usage
item_id = 14  # Replace with an actual item ID (recipe ID)
recommended_items = recommend_items(item_id, item_user_matrix_approx_df)
print(f"Recommended items for item {item_id}:\n", recommended_items)

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Sample data loading (adjust the path and format as needed)
# data = pd.read_csv("/content/data_with_sentiment.csv")

# Ensure data contains necessary columns
required_columns = ['recipe_id', 'user_id', 'star_rating']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Required columns are missing from the dataset.")

# Create the item-user matrix
item_user_matrix = data.pivot_table(index='recipe_id', columns='user_id', values='star_rating')

# Fill NaN values with 0 for SVD processing
item_user_matrix = item_user_matrix.fillna(0)

# Convert the matrix to a sparse matrix format
item_user_matrix_sparse = csr_matrix(item_user_matrix.values)

# Define the number of latent factors
n_factors = 20

# Apply SVD
svd = TruncatedSVD(n_components=n_factors)
item_user_matrix_svd = svd.fit_transform(item_user_matrix_sparse)

# Compute the approximate item-user matrix
item_user_matrix_approx = item_user_matrix_svd.dot(svd.components_)

# Convert the approximate matrix back to a DataFrame
item_user_matrix_approx_df = pd.DataFrame(item_user_matrix_approx,
                                          index=item_user_matrix.index,
                                          columns=item_user_matrix.columns)

# Function to get item recommendations
def recommend_items(item_id, item_user_matrix_approx_df, top_n=10):
    if item_id not in item_user_matrix_approx_df.index:
        raise ValueError(f"Item ID {item_id} not found in the matrix.")

    # Predict ratings for the given item
    item_ratings = item_user_matrix_approx_df.loc[item_id]

    # Get the top_n items with the highest predicted ratings
    recommendations = item_ratings.sort_values(ascending=False).head(top_n)

    return recommendations

# Example usage
item_id = 18
if item_id in item_user_matrix_approx_df.index:
    recommended_items = recommend_items(item_id, item_user_matrix_approx_df)
    print(f"Recommended items for item {item_id}:\n", recommended_items)
else:
    print(f"Item ID {item_id} does not exist in the data.")

# Function to get item-based recommendations
def recommend_similar_items(item_id, item_user_matrix_approx_df, top_n=10):
    if item_id not in item_user_matrix_approx_df.index:
        raise ValueError(f"Item ID {item_id} not found in the matrix.")

    # Predict ratings for the given item
    item_ratings = item_user_matrix_approx_df.loc[item_id]

    # Drop the item itself from recommendations
    item_ratings = item_ratings.drop(item_id, errors='ignore')

    # Get the top_n items with the highest predicted ratings
    recommendations = item_ratings.sort_values(ascending=False).head(top_n)

    return recommendations

# Example usage
item_id = 18  # Replace with an actual item ID (recipe ID)
if item_id in item_user_matrix_approx_df.index:
    recommended_items = recommend_similar_items(item_id, item_user_matrix_approx_df)
    print(f"Recommended items similar to item {item_id}:\n", recommended_items)
else:
    print(f"Item ID {item_id} does not exist in the data.")

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Sample data (replace this with your actual data)
# data = pd.read_csv("/content/data_with_sentiment.csv")

# Aggregate Duplicate Entries
df_aggregated = data.groupby(['user_id', 'recipe_id', 'recipe_name']).agg({'sentiment_score': 'mean'}).reset_index()

# Verify no duplicates
duplicates = df_aggregated[df_aggregated.duplicated(['user_id', 'recipe_id'], keep=False)]
print("Duplicates after aggregation:", duplicates)

# Create User-Item Interaction Matrix
user_item_matrix = df_aggregated.pivot_table(index='user_id', columns='recipe_id', values='sentiment_score', aggfunc='mean').fillna(0)

# Convert user-item matrix to a sparse matrix
user_item_sparse = csr_matrix(user_item_matrix.values)

# Inspect the shape of the user-item matrix
print("User-Item Matrix Shape:", user_item_matrix.shape)

# Choose an appropriate value for k
k = min(user_item_matrix.shape) - 1  # It should be less than the minimum dimension of the matrix

# Perform SVD
U, sigma, Vt = svds(user_item_sparse, k=k)  # Choose k latent factors

# Convert sigma to diagonal matrix form
sigma = np.diag(sigma)

# Reconstruct the approximate user-item matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

def recommend_recipes_svd(user_id, original_ratings_matrix, predicted_ratings_matrix, num_recommendations=5):
    if user_id not in original_ratings_matrix.index:
        raise ValueError(f"User ID {user_id} not found in the matrix.")

    user_idx = original_ratings_matrix.index.get_loc(user_id)
    sorted_user_predictions = predicted_ratings_matrix.iloc[user_idx].sort_values(ascending=False)
    user_data = original_ratings_matrix.loc[user_id]
    recommendations = sorted_user_predictions[~user_data.index.isin(user_data[user_data > 0].index)]
    return recommendations.head(num_recommendations).index.tolist()

# Example usage
user_id = 'u_1oKW8OaITme33Q0dR4Y4AhYSzkO'  # Replace with the user ID you want to recommend recipes for

if user_id in user_item_matrix.index:
    recommended_recipes_ids = recommend_recipes_svd(user_id, user_item_matrix, predicted_ratings_df)

    if not recommended_recipes_ids:
        print("No recommendations available for this user.")
    else:
        # Extract recommended recipe names from the original dataset
        recommended_recipes = df_aggregated[df_aggregated['recipe_id'].isin(recommended_recipes_ids)][['recipe_id', 'recipe_name']].drop_duplicates()
        print("Recommended Recipes Details:\n", recommended_recipes)

        # Remove duplicates from recommended recipes
        recommended_recipes_unique = recommended_recipes.drop_duplicates(subset=['recipe_id'])
        print("Recommended Recipes Details (Unique):\n", recommended_recipes_unique)
else:
    print(f"User ID {user_id} not found in the user-item matrix.")

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Load data (replace this with your actual data)
# data = pd.read_csv("/content/data_with_sentiment.csv")

# Aggregate Duplicate Entries
df_aggregated = data.groupby(['user_id', 'recipe_id', 'recipe_name']).agg({'sentiment_score': 'mean'}).reset_index()

# Verify no duplicates
duplicates = df_aggregated[df_aggregated.duplicated(['user_id', 'recipe_id'], keep=False)]
print("Duplicates after aggregation:", duplicates)

# Create User-Item Interaction Matrix
user_item_matrix = df_aggregated.pivot_table(index='user_id', columns='recipe_id', values='sentiment_score', aggfunc='mean').fillna(0)

# Convert user-item matrix to a sparse matrix
user_item_sparse = csr_matrix(user_item_matrix.values)

# Inspect the shape of the user-item matrix
print("User-Item Matrix Shape:", user_item_matrix.shape)

# Choose an appropriate value for k
k = min(user_item_matrix.shape) - 1

# Perform SVD
U, sigma, Vt = svds(user_item_sparse, k=k)

# Convert sigma to diagonal matrix form
sigma = np.diag(sigma)

# Reconstruct the approximate user-item matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

def recommend_recipes_svd(user_id, original_ratings_matrix, predicted_ratings_matrix, num_recommendations=20):
    if user_id not in original_ratings_matrix.index:
        raise ValueError(f"User ID {user_id} not found in the matrix.")

    user_idx = original_ratings_matrix.index.get_loc(user_id)
    sorted_user_predictions = predicted_ratings_matrix.iloc[user_idx].sort_values(ascending=False)
    user_data = original_ratings_matrix.loc[user_id]
    recommendations = sorted_user_predictions[~user_data.index.isin(user_data[user_data > 0].index)]
    return recommendations.head(num_recommendations).index.tolist()

# Example usage
user_id = 'u_1oKW8OaITme33Q0dR4Y4AhYSzkO'  # Replace with the user ID you want to recommend recipes for

if user_id in user_item_matrix.index:
    recommended_recipes_ids = recommend_recipes_svd(user_id, user_item_matrix, predicted_ratings_df)

    if not recommended_recipes_ids:
        print("No recommendations available for this user.")
    else:
        # Extract recommended recipe names from the original dataset
        recommended_recipes = df_aggregated[df_aggregated['recipe_id'].isin(recommended_recipes_ids)][['recipe_id', 'recipe_name']].drop_duplicates()
        print("Recommended Recipes Details:\n", recommended_recipes)

        # Remove duplicates from recommended recipes
        recommended_recipes_unique = recommended_recipes.drop_duplicates(subset=['recipe_id'])
        print("Recommended Recipes Details (Unique):\n", recommended_recipes_unique)
else:
    print(f"User ID {user_id} not found in the user-item matrix.")

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Sample data (replace this with your actual data)
#data = pd.read_csv("/content/data_with_sentiment.csv")

# Aggregate Duplicate Entries
df_aggregated = data.groupby(['user_id', 'recipe_id', 'recipe_name', 'processed_review_text', 'sentiment']).agg({'sentiment_score': 'mean'}).reset_index()

# Verify no duplicates
duplicates = df_aggregated[df_aggregated.duplicated(['user_id', 'recipe_id'], keep=False)]
print("Duplicates after aggregation:", duplicates)

# Create User-Item Interaction Matrix
user_item_matrix = df_aggregated.pivot_table(index='user_id', columns='recipe_id', values='sentiment_score', aggfunc='mean').fillna(0)

# Convert user-item matrix to a sparse matrix
user_item_sparse = csr_matrix(user_item_matrix.values)

# Inspect the shape of the user-item matrix
print("User-Item Matrix Shape:", user_item_matrix.shape)

# Choose an appropriate value for k
k = min(user_item_matrix.shape) - 1  # It should be less than the minimum dimension of the matrix

# Perform SVD
U, sigma, Vt = svds(user_item_sparse, k=k)  # Choose k latent factors

# Convert sigma to diagonal matrix form
sigma = np.diag(sigma)

# Reconstruct the approximate user-item matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

def recommend_recipes_svd(user_id, original_ratings_matrix, predicted_ratings_matrix, num_recommendations=5):
    user_idx = original_ratings_matrix.index.get_loc(user_id)
    sorted_user_predictions = predicted_ratings_matrix.iloc[user_idx].sort_values(ascending=False)
    user_data = original_ratings_matrix.loc[user_id]
    recommendations = sorted_user_predictions[~user_data.index.isin(user_data[user_data > 0].index)]
    return recommendations.head(num_recommendations).index.tolist()

# Example usage
user_id = 'u_1oKW8OaITme33Q0dR4Y4AhYSzkO'  # Replace with the user ID you want to recommend recipes for
recommended_recipes_ids = recommend_recipes_svd(user_id, user_item_matrix, predicted_ratings_df)
print("Recommended Recipe IDs:", recommended_recipes_ids)

# Extract recommended recipe names from the original dataset
recommended_recipes = df_aggregated[df_aggregated['recipe_id'].isin(recommended_recipes_ids)][['recipe_id', 'recipe_name', 'processed_review_text', 'sentiment', 'sentiment_score']].drop_duplicates()
print("Recommended Recipes Details:\n", recommended_recipes)
recommended_recipes

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import ipywidgets as widgets
from IPython.display import display

# Sample data (replace this with your actual data)
# data = pd.read_csv("/content/data_with_sentiment.csv")

# Aggregate Duplicate Entries
df_aggregated = data.groupby(['user_id', 'recipe_id', 'recipe_name', 'processed_review_text', 'sentiment']).agg({'sentiment_score': 'mean'}).reset_index()

# Verify no duplicates
duplicates = df_aggregated[df_aggregated.duplicated(['user_id', 'recipe_id'], keep=False)]
print("Duplicates after aggregation:", duplicates)

# Create User-Item Interaction Matrix
user_item_matrix = df_aggregated.pivot_table(index='user_id', columns='recipe_id', values='sentiment_score', aggfunc='mean').fillna(0)

# Convert user-item matrix to a sparse matrix
user_item_sparse = csr_matrix(user_item_matrix.values)

# Inspect the shape of the user-item matrix
print("User-Item Matrix Shape:", user_item_matrix.shape)

# Choose an appropriate value for k
k = min(user_item_matrix.shape) - 1  # It should be less than the minimum dimension of the matrix

# Perform SVD
U, sigma, Vt = svds(user_item_sparse, k=k)  # Choose k latent factors

# Convert sigma to diagonal matrix form
sigma = np.diag(sigma)

# Reconstruct the approximate user-item matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

def recommend_recipes_svd(user_id, original_ratings_matrix, predicted_ratings_matrix, num_recommendations=5):
    user_idx = original_ratings_matrix.index.get_loc(user_id)
    sorted_user_predictions = predicted_ratings_matrix.iloc[user_idx].sort_values(ascending=False)
    user_data = original_ratings_matrix.loc[user_id]
    recommendations = sorted_user_predictions[~user_data.index.isin(user_data[user_data > 0].index)]
    return recommendations.head(num_recommendations).index.tolist()

# Dropdown widget for user ID selection
user_id_dropdown = widgets.Dropdown(
    options=user_item_matrix.index.tolist(),
    description='User ID:',
    disabled=False,
)

# Function to display recommendations on user selection
def on_user_selection(change):
    user_id = change['new']
    recommended_recipes_ids = recommend_recipes_svd(user_id, user_item_matrix, predicted_ratings_df)
    print("Recommended Recipe IDs:", recommended_recipes_ids)

    # Extract recommended recipe names from the original dataset
    recommended_recipes = df_aggregated[df_aggregated['recipe_id'].isin(recommended_recipes_ids)][['recipe_id', 'recipe_name', 'processed_review_text', 'sentiment', 'sentiment_score']].drop_duplicates()
    print("Recommended Recipes Details:\n", recommended_recipes)

# Observe changes in the dropdown and run the recommendation logic
user_id_dropdown.observe(on_user_selection, names='value')

# Display the dropdown
display(user_id_dropdown)

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import ipywidgets as widgets
from IPython.display import display

# Sample data (replace this with your actual data)
# data = pd.read_csv("/content/data_with_sentiment.csv")

# Aggregate Duplicate Entries
df_aggregated = data.groupby(['user_id', 'recipe_id', 'recipe_name', 'processed_review_text', 'sentiment']).agg({'sentiment_score': 'mean'}).reset_index()

# Verify no duplicates
duplicates = df_aggregated[df_aggregated.duplicated(['user_id', 'recipe_id'], keep=False)]
print("Duplicates after aggregation:", duplicates)

# Create User-Item Interaction Matrix
user_item_matrix = df_aggregated.pivot_table(index='user_id', columns='recipe_id', values='sentiment_score', aggfunc='mean').fillna(0)

# Convert user-item matrix to a sparse matrix
user_item_sparse = csr_matrix(user_item_matrix.values)

# Inspect the shape of the user-item matrix
print("User-Item Matrix Shape:", user_item_matrix.shape)

# Choose an appropriate value for k
k = min(user_item_matrix.shape) - 1  # It should be less than the minimum dimension of the matrix

# Perform SVD
U, sigma, Vt = svds(user_item_sparse, k=k)  # Choose k latent factors

# Convert sigma to diagonal matrix form
sigma = np.diag(sigma)

# Reconstruct the approximate user-item matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

def recommend_recipes_svd(user_id, original_ratings_matrix, predicted_ratings_matrix, num_recommendations=5):
    user_idx = original_ratings_matrix.index.get_loc(user_id)
    sorted_user_predictions = predicted_ratings_matrix.iloc[user_idx].sort_values(ascending=False)
    user_data = original_ratings_matrix.loc[user_id]
    recommendations = sorted_user_predictions[~user_data.index.isin(user_data[user_data > 0].index)]
    return recommendations.head(num_recommendations).index.tolist()

# Dropdown widget for user ID selection
user_id_dropdown = widgets.Dropdown(
    options=user_item_matrix.index.tolist(),
    description='User ID:',
    disabled=False,
)

# Function to display recommendations on user selection
def on_user_selection(change):
    user_id = change['new']
    print(f"Recommendations for User ID: {user_id}")

    recommended_recipes_ids = recommend_recipes_svd(user_id, user_item_matrix, predicted_ratings_df)

    if not recommended_recipes_ids:
        print(f"No recommendations available for User ID: {user_id}")
    else:
        # Extract recommended recipe names from the original dataset
        recommended_recipes = df_aggregated[df_aggregated['recipe_id'].isin(recommended_recipes_ids)][['recipe_id', 'recipe_name', 'processed_review_text', 'sentiment', 'sentiment_score']].drop_duplicates()
        display(recommended_recipes)

# Observe changes in the dropdown and run the recommendation logic
user_id_dropdown.observe(on_user_selection, names='value')

# Display the dropdown
display(user_id_dropdown)

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import ipywidgets as widgets
from IPython.display import display

# Sample data (replace this with your actual data)
# data = pd.read_csv("/content/data_with_sentiment.csv")

# Filter for positive sentiment reviews
data = data[data['sentiment_score'] > 0]

# Aggregate Duplicate Entries
df_aggregated = data.groupby(['user_id', 'recipe_id', 'recipe_name', 'processed_review_text', 'sentiment']).agg({'sentiment_score': 'mean'}).reset_index()

# Verify no duplicates
duplicates = df_aggregated[df_aggregated.duplicated(['user_id', 'recipe_id'], keep=False)]
print("Duplicates after aggregation:", duplicates)

# Create User-Item Interaction Matrix
user_item_matrix = df_aggregated.pivot_table(index='user_id', columns='recipe_id', values='sentiment_score', aggfunc='mean').fillna(0)

# Convert user-item matrix to a sparse matrix
user_item_sparse = csr_matrix(user_item_matrix.values)

# Inspect the shape of the user-item matrix
print("User-Item Matrix Shape:", user_item_matrix.shape)

# Choose an appropriate value for k
k = min(user_item_matrix.shape) - 1  # It should be less than the minimum dimension of the matrix

# Perform SVD
U, sigma, Vt = svds(user_item_sparse, k=k)  # Choose k latent factors

# Convert sigma to diagonal matrix form
sigma = np.diag(sigma)

# Reconstruct the approximate user-item matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

def recommend_recipes_svd(user_id, original_ratings_matrix, predicted_ratings_matrix, num_recommendations=5):
    user_idx = original_ratings_matrix.index.get_loc(user_id)
    sorted_user_predictions = predicted_ratings_matrix.iloc[user_idx].sort_values(ascending=False)
    user_data = original_ratings_matrix.loc[user_id]
    recommendations = sorted_user_predictions[~user_data.index.isin(user_data[user_data > 0].index)]
    return recommendations.head(num_recommendations).index.tolist()

# Dropdown widget for user ID selection
user_id_dropdown = widgets.Dropdown(
    options=user_item_matrix.index.tolist(),
    description='User ID:',
    disabled=False,
)

# Function to display recommendations on user selection
def on_user_selection(change):
    user_id = change['new']
    print(f"Recommendations for User ID: {user_id}")

    recommended_recipes_ids = recommend_recipes_svd(user_id, user_item_matrix, predicted_ratings_df)

    if not recommended_recipes_ids:
        print(f"No recommendations available for User ID: {user_id}")
    else:
        # Extract recommended recipe names with positive sentiment from the original dataset
        recommended_recipes = df_aggregated[
            (df_aggregated['recipe_id'].isin(recommended_recipes_ids)) & (df_aggregated['sentiment_score'] > 0)
        ][['recipe_id', 'recipe_name', 'processed_review_text', 'sentiment', 'sentiment_score']].drop_duplicates()

        if recommended_recipes.empty:
            print(f"No positive sentiment recommendations available for User ID: {user_id}")
        else:
            display(recommended_recipes)

# Observe changes in the dropdown and run the recommendation logic
user_id_dropdown.observe(on_user_selection, names='value')

# Display the dropdown
display(user_id_dropdown)

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import ipywidgets as widgets
from IPython.display import display

# Sample data (replace this with your actual data)
# data = pd.read_csv("/content/data_with_sentiment.csv")

# Filter for positive sentiment reviews
data = data[data['sentiment_score'] > 0]

# Aggregate Duplicate Entries (including star_rating)
df_aggregated = data.groupby(
    ['user_id', 'recipe_id', 'recipe_name', 'processed_review_text', 'sentiment']
).agg({
    'sentiment_score': 'mean',
    'star_rating': 'mode'  # Aggregate star_rating as well
}).reset_index()

# Verify no duplicates
duplicates = df_aggregated[df_aggregated.duplicated(['user_id', 'recipe_id'], keep=False)]
print("Duplicates after aggregation:", duplicates)

# Create User-Item Interaction Matrix (use sentiment_score for interaction values)
user_item_matrix = df_aggregated.pivot_table(index='user_id', columns='recipe_id', values='sentiment_score', aggfunc='mean').fillna(0)

# Convert user-item matrix to a sparse matrix
user_item_sparse = csr_matrix(user_item_matrix.values)

# Inspect the shape of the user-item matrix
print("User-Item Matrix Shape:", user_item_matrix.shape)

# Choose an appropriate value for k
k = min(user_item_matrix.shape) - 1  # It should be less than the minimum dimension of the matrix

# Perform SVD
U, sigma, Vt = svds(user_item_sparse, k=k)  # Choose k latent factors

# Convert sigma to diagonal matrix form
sigma = np.diag(sigma)

# Reconstruct the approximate user-item matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Recommendation function
def recommend_recipes_svd(user_id, original_ratings_matrix, predicted_ratings_matrix, num_recommendations=5):
    user_idx = original_ratings_matrix.index.get_loc(user_id)
    sorted_user_predictions = predicted_ratings_matrix.iloc[user_idx].sort_values(ascending=False)

    user_data = original_ratings_matrix.loc[user_id]
    recommendations = sorted_user_predictions[~user_data.index.isin(user_data[user_data > 0].index)]

    return recommendations.head(num_recommendations).index.tolist()

# Dropdown widget for user ID selection
user_id_dropdown = widgets.Dropdown(
    options=user_item_matrix.index.tolist(),
    description='User ID:',
    disabled=False,
)

# Function to display recommendations on user selection
def on_user_selection(change):
    user_id = change['new']
    print(f"Recommendations for User ID: {user_id}")

    recommended_recipes_ids = recommend_recipes_svd(user_id, user_item_matrix, predicted_ratings_df)

    if not recommended_recipes_ids:
        print(f"No recommendations available for User ID: {user_id}")
    else:
        # Extract recommended recipe names and star ratings from the original dataset
        recommended_recipes = df_aggregated[
            (df_aggregated['recipe_id'].isin(recommended_recipes_ids)) & (df_aggregated['sentiment_score'] > 0)
        ][['recipe_id', 'recipe_name', 'star_rating', 'processed_review_text', 'sentiment', 'sentiment_score']].drop_duplicates()

        if recommended_recipes.empty:
            print(f"No positive sentiment recommendations available for User ID: {user_id}")
        else:
            display(recommended_recipes)

# Observe changes in the dropdown and run the recommendation logic
user_id_dropdown.observe(on_user_selection, names='value')

# Display the dropdown
display(user_id_dropdown)

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import ipywidgets as widgets
from IPython.display import display

# Sample data (replace this with your actual data)
# data = pd.read_csv("/content/data_with_sentiment.csv")

# Filter for positive sentiment reviews
data = data[data['sentiment_score'] > 0]

# Filter out negative star ratings
recommendations = recommendations[recommendations['star_rating'] > 0]

# No need to aggregate the star_rating
df_aggregated = data.drop_duplicates(subset=['user_id', 'recipe_id'])

# Verify no duplicates
duplicates = df_aggregated[df_aggregated.duplicated(['user_id', 'recipe_id'], keep=False)]
print("Duplicates after aggregation:", duplicates)

# Create User-Item Interaction Matrix (use sentiment_score for interaction values)
user_item_matrix = df_aggregated.pivot_table(index='user_id', columns='recipe_id', values='sentiment_score', aggfunc='mean').fillna(0)

# Convert user-item matrix to a sparse matrix
user_item_sparse = csr_matrix(user_item_matrix.values)

# Inspect the shape of the user-item matrix
print("User-Item Matrix Shape:", user_item_matrix.shape)

# Choose an appropriate value for k
k = min(user_item_matrix.shape) - 1  # It should be less than the minimum dimension of the matrix

# Perform SVD
U, sigma, Vt = svds(user_item_sparse, k=k)  # Choose k latent factors

# Convert sigma to diagonal matrix form
sigma = np.diag(sigma)

# Reconstruct the approximate user-item matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Recommendation function
def recommend_recipes_svd(user_id, original_ratings_matrix, predicted_ratings_matrix, num_recommendations=5):
    user_idx = original_ratings_matrix.index.get_loc(user_id)
    sorted_user_predictions = predicted_ratings_matrix.iloc[user_idx].sort_values(ascending=False)

    user_data = original_ratings_matrix.loc[user_id]
    recommendations = sorted_user_predictions[~user_data.index.isin(user_data[user_data > 0].index)]

    return recommendations.head(num_recommendations).index.tolist()

# Dropdown widget for user ID selection
user_id_dropdown = widgets.Dropdown(
    options=user_item_matrix.index.tolist(),
    description='User ID:',
    disabled=False,
)

# Function to save recommendations to CSV based on selected user
def save_recommendations_to_csv(recommended_recipes, user_id):
    # Save to CSV (update path as needed)
    csv_filename = f'recommendations_for_user_{user_id}.csv'
    recommended_recipes.to_csv(csv_filename, index=False)
    print(f"Recommendations saved to {csv_filename}")

# Function to display recommendations and save to CSV on user selection
def on_user_selection(change):
    user_id = change['new']
    print(f"Generating recommendations for User ID: {user_id}")

    recommended_recipes_ids = recommend_recipes_svd(user_id, user_item_matrix, predicted_ratings_df)

    if not recommended_recipes_ids:
        print(f"No recommendations available for User ID: {user_id}")
    else:
        # Extract recommended recipe names and star ratings from the original dataset
        recommended_recipes = df_aggregated[
            (df_aggregated['recipe_id'].isin(recommended_recipes_ids)) & (df_aggregated['sentiment_score'] > 0.5)
        ][['recipe_id', 'recipe_name', 'star_rating', 'processed_review_text', 'sentiment', 'sentiment_score']].drop_duplicates()

        if recommended_recipes.empty:
            print(f"No positive sentiment recommendations available for User ID: {user_id}")
        else:
            display(recommended_recipes)
            save_recommendations_to_csv(recommended_recipes, user_id)

# Observe changes in the dropdown and run the recommendation logic
user_id_dropdown.observe(on_user_selection, names='value')

# Display the dropdown
display(user_id_dropdown)

data

"""# **Multinomial Naive Bayes**"""

# Example code for sentiment classification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler

# Ensure that all text data is string type and replace NaN values
data['processed_review_text'] = data['processed_review_text'].astype(str)

# Scale numerical features to be non-negative (only if needed)
scaler = MinMaxScaler()
scaled_numerical_features = scaler.fit_transform(data[['user_reputation', 'thumbs_up', 'thumbs_down', 'star_rating', 'best_score']])

# Prepare features and target variable
X = data['processed_review_text']
y = data['sentiment']

# Vectorize the review text
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(X)

# Combine numerical features with text features
X_combined = hstack([scaled_numerical_features, X_text])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

# Train a model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack

# Load your data
# data = pd.read_csv('your_data.csv')  # Replace with your data loading method

# Ensure that all text data is string type and replace NaN values
data['processed_review_text'] = data['processed_review_text'].astype(str)

# Scale numerical features
scaler = MinMaxScaler()
scaled_numerical_features = scaler.fit_transform(data[['user_reputation', 'thumbs_up', 'thumbs_down', 'star_rating', 'best_score']])

# Prepare features and target variable
X_text = data['processed_review_text']
y = data['sentiment']

# Vectorize the review text
vectorizer = TfidfVectorizer()
X_text_vectorized = vectorizer.fit_transform(X_text)

# Combine numerical features with text features
X_combined = hstack([scaled_numerical_features, X_text_vectorized])

# Feature Selection (optional)
selector = SelectKBest(chi2, k=1000)  # Select top 1000 features
X_selected = selector.fit_transform(X_combined, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Define and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_selected, y, cv=5)  # 5-fold cross-validation
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean()}')

# Make predictions and evaluate
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.pipeline import Pipeline as ImbPipeline

# Ensure that all text data is string type
data['processed_review_text'] = data['processed_review_text'].astype(str)

# Prepare features and target variable
X = data[['processed_review_text', 'user_reputation', 'thumbs_up', 'thumbs_down', 'star_rating', 'best_score']]
y = data['sentiment']

# Vectorize the review text
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(X['processed_review_text'])

# Scale numerical features
scaler = MinMaxScaler()
scaled_numerical_features = scaler.fit_transform(X[['user_reputation', 'thumbs_up', 'thumbs_down', 'star_rating', 'best_score']])

# Combine numerical features with text features
X_combined = hstack([scaled_numerical_features, X_text])

# Feature selection
selector = SelectKBest(chi2, k=1000)  # Adjust the number of features as needed
X_selected = selector.fit_transform(X_combined, y)

# Resample the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
class_weight_dict = dict(zip(np.unique(y_resampled), class_weights))

# Define the Naive Bayes model with class weights
model = MultinomialNB()

# Define the parameter grid for hyperparameter tuning
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}

# Initialize the GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')

# Fit the grid search to the data
grid_search.fit(X_resampled, y_resampled)

# Best parameters and score
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_}')

# Train the model with the best parameters
model = grid_search.best_estimator_

# Split the data again
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Make predictions and evaluate
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Generate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, class_names=['Negative', 'Neutral', 'Positive'])

"""# Recurrent Neural Networks (RNNs) for Sequential Recommendations"""



import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your dataset
#df = pd.read_csv('path_to_your_data.csv')

# Extract features and target
X = data[['review_text']]
y = data['sentiment_score']

# Tokenize text data
tokenizer = Tokenizer(num_words=10000)  # Adjust num_words as needed
tokenizer.fit_on_texts(X['review_text'])
X_sequences = tokenizer.texts_to_sequences(X['review_text'])

# Pad sequences to ensure uniform length
max_sequence_length = 100  # Adjust based on your data
X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.3, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test MAE: {mae}')
# Prepare new data
new_reviews = ["This recipe was fantastic!", "Not great, too salty."]

# Tokenize and pad new text data
new_X_sequences = tokenizer.texts_to_sequences(new_reviews)
new_X_padded = pad_sequences(new_X_sequences, maxlen=max_sequence_length)

# Predict sentiment scores
predicted_scores = model.predict(new_X_padded)
print(predicted_scores)

data

"""# Multi-Task Neural Network for Sentiment Classification and Regression
Dual-Objective Neural Network for Sentiment and Rating Prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Concatenate, Input, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
# df = pd.read_csv('your_dataset.csv')

# Select target variable and features
X_text = data['processed_review_text'].astype(str).fillna('')
X_numerical = data[['user_reputation', 'thumbs_up', 'thumbs_down', 'star_rating', 'best_score']]
y_classification = pd.get_dummies(data['sentiment']).values  # Classification target
y_regression = data['sentiment_score']  # Regression target

# Text preprocessing
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_text)
X_text_seq = tokenizer.texts_to_sequences(X_text)
X_text_pad = pad_sequences(X_text_seq, maxlen=max_len)

# Standardize numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Split the data
X_train_text, X_test_text, X_train_num, X_test_num, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X_text_pad, X_numerical_scaled, y_classification, y_regression, test_size=0.3, random_state=42
)

# LSTM model for text data
input_text = Input(shape=(max_len,))
embedding = Embedding(input_dim=max_words, output_dim=128)(input_text)
lstm_out = LSTM(128)(embedding)

# Combine LSTM output with numerical features
input_numerical = Input(shape=(X_numerical.shape[1],))
combined = Concatenate()([lstm_out, input_numerical])

# Dense layers after combining
dense = Dense(64, activation='relu')(combined)
dropout = Dropout(0.5)(dense)

# Output layers
output_class = Dense(y_classification.shape[1], activation='softmax', name='classification_output')(dropout)
output_reg = Dense(1, name='regression_output')(dropout)

# Define the model
model = Model(inputs=[input_text, input_numerical], outputs=[output_class, output_reg])

# Compile the model with different loss functions and metrics for each output
model.compile(
    loss={'classification_output': 'categorical_crossentropy', 'regression_output': 'mean_squared_error'},
    optimizer='adam',
    metrics={'classification_output': 'accuracy', 'regression_output': 'mean_squared_error'}
)

# Train the model
history = model.fit(
    [X_train_text, X_train_num],
    {'classification_output': y_train_class, 'regression_output': y_train_reg},
    epochs=3,
    batch_size=32,
    validation_split=0.2
)

# Evaluate the model
metrics = model.evaluate([X_test_text, X_test_num],
                         {'classification_output': y_test_class, 'regression_output': y_test_reg},
                         verbose=0)

# Debug: Check the length of metrics
print(f"Metrics list length: {len(metrics)}")

# Adjusted metric printing
if len(metrics) == 3:
    print(f"Test Loss (Overall): {metrics[0]}")
    print(f"Test Accuracy (Classification): {metrics[1]}")
    print(f"Test Loss (Regression): {metrics[2]}")
elif len(metrics) == 5:
    print(f"Test Loss (Overall): {metrics[0]}")
    print(f"Test Loss (Classification): {metrics[1]}")
    print(f"Test Accuracy (Classification): {metrics[2]}")
    print(f"Test Loss (Regression): {metrics[3]}")
    print(f"Test MSE (Regression): {metrics[4]}")
else:
    print("Unexpected metrics length. Please check the model configuration.")

# Predict sentiment and sentiment score for new reviews
new_reviews = ["This recipe was amazing! I loved it.", "The dish was too salty for my taste."]
new_reviews_seq = tokenizer.texts_to_sequences(new_reviews)
new_reviews_pad = pad_sequences(new_reviews_seq, maxlen=max_len)
new_numerical_data = np.array([[0.5, 10, 2, 5, 3], [0.2, 5, 1, 2, 2]])  # Example numerical data for the new reviews

pred_class, pred_reg = model.predict([new_reviews_pad, new_numerical_data])

# Convert classification predictions to sentiment labels
sentiment_labels = ['Negative', 'Neutral', 'Positive']
predicted_sentiments = [sentiment_labels[np.argmax(pred)] for pred in pred_class]

# Print predictions
for i, review in enumerate(new_reviews):
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {predicted_sentiments[i]}")
    print(f"Predicted Sentiment Score: {pred_reg[i][0]}")
    print("-" * 50)

"""# **Text Processing and Sentiment Score Prediction Model**
Integrated Model for Sentiment Classification and Sentiment Score Prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Ensure that all text data is string type and replace NaN values
data['processed_review_text'] = data['processed_review_text'].astype(str).fillna('')

# Preprocessing
max_words = 10000  # Maximum number of words to consider in the tokenizer
max_len = 100  # Maximum length of sequences

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data['processed_review_text'])
X_text_seq = tokenizer.texts_to_sequences(data['processed_review_text'])
X_text_pad = pad_sequences(X_text_seq, maxlen=max_len)

# Select numerical features
X_numerical = data[['user_reputation', 'thumbs_up', 'thumbs_down', 'star_rating', 'best_score']].values

# Combine text features with numerical features
X_combined = np.hstack([X_text_pad, X_numerical])

# Encode target variable (Sentiment)
y_sentiment = pd.get_dummies(data['sentiment']).values

# Regression target (Star Rating)
y_star_rating = data['star_rating'].values

# Train-test split
X_train, X_test, y_train_sentiment, y_test_sentiment, y_train_star, y_test_star = train_test_split(
    X_combined, y_sentiment, y_star_rating, test_size=0.3, random_state=42)

# Define the model architecture
input_text = Input(shape=(max_len,))
input_numerical = Input(shape=(X_numerical.shape[1],))

# Text Processing (Embedding + LSTM)
x = Embedding(input_dim=max_words, output_dim=128)(input_text)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Bidirectional(LSTM(32))(x)

# Combine with Dense layers for both tasks
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)

# Concatenate text and numerical features
combined = Dense(64, activation='relu')(input_numerical)
combined = Dropout(0.5)(combined)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# Classification output for sentiment analysis
classification_output = Dense(y_sentiment.shape[1], activation='softmax', name='classification_output')(x)

# Regression output for star rating prediction
regression_output = Dense(1, name='regression_output')(combined)

# Define the model
model = Model(inputs=[input_text, input_numerical], outputs=[classification_output, regression_output])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for better learning
    loss={'classification_output': 'categorical_crossentropy', 'regression_output': 'mean_squared_error'},
    metrics={'classification_output': 'accuracy', 'regression_output': 'mean_squared_error'}
)

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    [X_train[:, :max_len], X_train[:, max_len:]],
    {'classification_output': y_train_sentiment, 'regression_output': y_train_star},
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluate the model
metrics = model.evaluate([X_test[:, :max_len], X_test[:, max_len:]], {'classification_output': y_test_sentiment, 'regression_output': y_test_star})
print(f"Test Loss (Overall): {metrics[0]}")
print(f"Test Accuracy (Classification): {metrics[1]}")
print(f"Test MSE (Regression): {metrics[2]}")

# Predict sentiment and sentiment score for new reviews
new_reviews = ["This recipe was amazing! I loved it.", "The dish was too salty for my taste."]

# Preprocess new reviews
new_reviews_seq = tokenizer.texts_to_sequences(new_reviews)
new_reviews_pad = pad_sequences(new_reviews_seq, maxlen=max_len)
new_reviews_combined = np.hstack([new_reviews_pad, np.zeros((len(new_reviews), X_numerical.shape[1]))])  # Assuming no numerical features for new reviews

# Make predictions
predictions = model.predict([new_reviews_pad, np.zeros((len(new_reviews), X_numerical.shape[1]))])
predicted_sentiment = np.argmax(predictions[0], axis=1)
predicted_score = predictions[1]

# Print predictions
for review, sentiment, score in zip(new_reviews, predicted_sentiment, predicted_score):
    sentiment_label = 'Positive' if sentiment == 2 else 'Negative' if sentiment == 1 else 'Negative'
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {sentiment_label}")
    print(f"Predicted Sentiment Score: {score}")
    print("--------------------------------------------------")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Load and prepare your dataset
#data = pd.read_csv('/content/data_with_sentiment.csv')

# Feature extraction
X_text = data['processed_review_text'].astype(str).fillna('')
X_numerical = data[['user_reputation', 'thumbs_up', 'thumbs_down', 'star_rating', 'best_score']]

# Text vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10000)
X_text_vec = vectorizer.fit_transform(X_text).toarray()

# Combine text features with numerical features
X = np.hstack([X_text_vec, X_numerical])

# Prepare targets
y_classification = data['sentiment']  # For classification
y_regression = data['sentiment_score']  # For regression

# Split data for classification and regression tasks
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.3, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.3, random_state=42)

# Train and evaluate Logistic Regression (classification task)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_class, y_train_class)
log_reg_predictions = log_reg.predict(X_test_class)
log_reg_accuracy = accuracy_score(y_test_class, log_reg_predictions)
print(f"Logistic Regression Accuracy: {log_reg_accuracy:.4f}")

# Train and evaluate Support Vector Regression (regression task)
svr = SVR(kernel='rbf')
svr.fit(X_train_reg, y_train_reg)
svr_predictions = svr.predict(X_test_reg)
svr_mse = mean_squared_error(y_test_reg, svr_predictions)
print(f"Support Vector Regression Mean Squared Error: {svr_mse:.4f}")

# Train and evaluate Random Forest Regressor (regression task)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)
rf_predictions = rf_regressor.predict(X_test_reg)
rf_mse = mean_squared_error(y_test_reg, rf_predictions)
print(f"Random Forest Regressor Mean Squared Error: {rf_mse:.4f}")

# Evaluate the neural network
nn_metrics = nn_model.evaluate(X_test, {'classification_output': y_test_sentiment, 'regression_output': y_test_star})
print(f"Neural Network Test Loss: {nn_metrics[0]:.4f}")
print(f"Neural Network Classification Accuracy: {nn_metrics[1]:.4f}")
print(f"Neural Network Regression MSE: {nn_metrics[2]:.4f}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, Bidirectional, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and prepare your dataset
# data = pd.read_csv('/content/data_with_sentiment.csv')

# Ensure that all text data is string type and replace NaN values
data['processed_review_text'] = data['processed_review_text'].astype(str).fillna('')

# Preprocessing
max_words = 10000  # Maximum number of words to consider in the tokenizer
max_len = 100  # Maximum length of sequences

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data['processed_review_text'])
X_text_seq = tokenizer.texts_to_sequences(data['processed_review_text'])
X_text_pad = pad_sequences(X_text_seq, maxlen=max_len)

# Select numerical features
X_numerical = data[['user_reputation', 'thumbs_up', 'thumbs_down', 'star_rating', 'best_score']].values

# Encode target variable (Sentiment)
y_sentiment = pd.get_dummies(data['sentiment']).values

# Regression target (Star Rating)
y_star_rating = data['star_rating'].values

# Train-test split for separate inputs
X_train_text, X_test_text, X_train_numerical, X_test_numerical, y_train_sentiment, y_test_sentiment, y_train_star, y_test_star = train_test_split(
    X_text_pad, X_numerical, y_sentiment, y_star_rating, test_size=0.3, random_state=42)

# Define the model architecture
input_text = Input(shape=(max_len,))
input_numerical = Input(shape=(X_numerical.shape[1],))

# Text Processing (Embedding + LSTM)
x = Embedding(input_dim=max_words, output_dim=128)(input_text)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Bidirectional(LSTM(32))(x)

# Dense layer after LSTM
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)

# Combine text and numerical inputs
combined = concatenate([x, input_numerical])
combined = Dense(64, activation='relu')(combined)
combined = Dropout(0.5)(combined)

# Classification output for sentiment analysis
classification_output = Dense(y_sentiment.shape[1], activation='softmax', name='classification_output')(combined)

# Regression output for star rating prediction
regression_output = Dense(1, name='regression_output')(combined)

# Define the model
model = Model(inputs=[input_text, input_numerical], outputs=[classification_output, regression_output])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for better learning
    loss={'classification_output': 'categorical_crossentropy', 'regression_output': 'mean_squared_error'},
    metrics={'classification_output': 'accuracy', 'regression_output': 'mean_squared_error'}
)

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    [X_train_text, X_train_numerical],
    {'classification_output': y_train_sentiment, 'regression_output': y_train_star},
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluate the neural network
nn_metrics = model.evaluate([X_test_text, X_test_numerical],
                            {'classification_output': y_test_sentiment, 'regression_output': y_test_star})
print(f"Neural Network Test Loss: {nn_metrics[0]:.4f}")
print(f"Neural Network Classification Accuracy: {nn_metrics[1]:.4f}")
print(f"Neural Network Regression MSE: {nn_metrics[2]:.4f}")

# Predict sentiment and sentiment score for new reviews
new_reviews = ["This recipe was amazing! I loved it.", "The dish was too salty for my taste."]

# Preprocess new reviews
new_reviews_seq = tokenizer.texts_to_sequences(new_reviews)
new_reviews_pad = pad_sequences(new_reviews_seq, maxlen=max_len)

# Assuming no numerical features for new reviews, set them as zeros
new_reviews_combined = np.zeros((len(new_reviews), X_numerical.shape[1]))

# Make predictions
predictions = model.predict([new_reviews_pad, new_reviews_combined])
predicted_sentiment = np.argmax(predictions[0], axis=1)
predicted_score = predictions[1]

# Print predictions
for review, sentiment, score in zip(new_reviews, predicted_sentiment, predicted_score):
    sentiment_label = 'Positive' if sentiment == 2 else 'Neutral' if sentiment == 1 else 'Negative'
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {sentiment_label}")
    print(f"Predicted Sentiment Score: {score}")
    print("--------------------------------------------------")

# Classical Models for Comparison

# Text vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_text_vec = vectorizer.fit_transform(data['processed_review_text'].astype(str).fillna('')).toarray()

# Combine text features with numerical features
X_combined = np.hstack([X_text_vec, X_numerical])

# Prepare targets for classification and regression
y_classification = data['sentiment']  # For classification
y_regression = data['sentiment_score']  # For regression

# Split data for classification and regression tasks
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_combined, y_classification, test_size=0.3, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_combined, y_regression, test_size=0.3, random_state=42)

# Train and evaluate Logistic Regression (classification task)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_class, y_train_class)
log_reg_predictions = log_reg.predict(X_test_class)
log_reg_accuracy = accuracy_score(y_test_class, log_reg_predictions)
print(f"Logistic Regression Accuracy: {log_reg_accuracy:.4f}")

# Train and evaluate Support Vector Regression (regression task)
svr = SVR(kernel='rbf')
svr.fit(X_train_reg, y_train_reg)
svr_predictions = svr.predict(X_test_reg)
svr_mse = mean_squared_error(y_test_reg, svr_predictions)
print(f"Support Vector Regression Mean Squared Error: {svr_mse:.4f}")

# Train and evaluate Random Forest Regressor (regression task)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)
rf_predictions = rf_regressor.predict(X_test_reg)
rf_mse = mean_squared_error(y_test_reg, rf_predictions)
print(f"Random Forest Regressor Mean Squared Error: {rf_mse:.4f}")







import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Bidirectional, LSTM, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Load your data
#data = pd.read_csv('data_with_sentiment.csv')

# Print the column names to verify
print(data.columns)

# Ensure column names match and handle missing columns
data['processed_review_text'] = data['processed_review_text'].astype(str).fillna('')
data['sentiment_score'] = data['sentiment_score'].fillna(0)  # Fill missing sentiment score values if any

# Map sentiment to numeric values for classification
sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
data['sentiment_numeric'] = data['sentiment'].map(sentiment_mapping)

# Separate features and targets
X_text = data['processed_review_text'].values
X_numerical = data[['user_reputation', 'thumbs_up', 'thumbs_down', 'best_score']].values
y_sentiment = data['sentiment_numeric'].values
y_star = data['star_rating'].values

# Text vectorization and padding
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_text)
X_text_seq = tokenizer.texts_to_sequences(X_text)
X_text_pad = pad_sequences(X_text_seq, maxlen=max_len)

# Normalize numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Apply SMOTE only to the classification task
smote = SMOTE(random_state=42)
X_text_smote, y_sentiment_smote = smote.fit_resample(X_text_pad, y_sentiment)

# For regression, use original data
X_numerical_final = X_numerical_scaled
y_star_final = y_star

# Align the data based on the number of samples
min_samples = min(len(X_text_smote), len(X_numerical_final))
X_text_smote = X_text_smote[:min_samples]
X_numerical_final = X_numerical_final[:min_samples]
y_sentiment_smote = y_sentiment_smote[:min_samples]
y_star_final = y_star_final[:min_samples]

# Define the model
input_text = Input(shape=(max_len,))
input_numerical = Input(shape=(X_numerical.shape[1],))

x = Embedding(input_dim=max_words, output_dim=128)(input_text)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Bidirectional(LSTM(32))(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)

combined = concatenate([x, input_numerical])
combined = Dense(64, activation='relu')(combined)
combined = Dropout(0.5)(combined)

classification_output = Dense(3, activation='softmax', name='classification_output')(combined)
regression_output = Dense(1, name='regression_output')(combined)

model = Model(inputs=[input_text, input_numerical], outputs=[classification_output, regression_output])
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss={'classification_output': 'sparse_categorical_crossentropy', 'regression_output': 'mean_squared_error'},
    metrics={'classification_output': 'accuracy', 'regression_output': 'mean_squared_error'}
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    [X_text_smote, X_numerical_final],
    {'classification_output': y_sentiment_smote, 'regression_output': y_star_final},
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluate the model
nn_metrics = model.evaluate([X_text_smote, X_numerical_final], {'classification_output': y_sentiment_smote, 'regression_output': y_star_final})
print(f"Neural Network Test Loss: {nn_metrics[0]:.4f}")
print(f"Neural Network Classification Accuracy: {nn_metrics[1]:.4f}")
print(f"Neural Network Regression MSE: {nn_metrics[2]:.4f}")

data.to_csv('data_NN_model.csv', index=False)

data.head()

from google.colab import sheets
sheet = sheets.InteractiveSheet(df=data)

# @title sentiment vs user_reputation

from matplotlib import pyplot as plt
import seaborn as sns
figsize = (12, 1.2 * len(data['sentiment'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(data, x='user_reputation', y='sentiment', inner='box', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)

# @title recipe_id

from matplotlib import pyplot as plt
data['recipe_id'].plot(kind='hist', bins=20, title='recipe_id')
plt.gca().spines[['top', 'right',]].set_visible(False)



# @title sentiment vs recipe_id

from matplotlib import pyplot as plt
import seaborn as sns
figsize = (12, 1.2 * len(data['sentiment'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(data, x='recipe_id', y='sentiment', inner='box', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)

# @title sentiment

from matplotlib import pyplot as plt
import seaborn as sns
data.groupby('sentiment').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

