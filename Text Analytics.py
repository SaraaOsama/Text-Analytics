#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import nltk
import spacy
import numpy as np


# In[42]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Word2Vec


# In[43]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[44]:


csv_file_path = "Restaurant_Reviews.csv"
df = pd.read_csv(csv_file_path)


# In[45]:


df.head()


# In[46]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')


# In[47]:


def tokenize_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    stop_words.discard("not")
    return [word for word in tokens if word.isalnum() and word not in stop_words]


# In[48]:


df["Review"] = df["Review"].apply(tokenize_text)
df.head()


# In[49]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[50]:


def lemmatize_dataframe(df):
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Initialize an empty list to store lemmatized words
    lemmatized_words = []

    # Process each row of the DataFrame
    for row in df["Review"]:
        # Initialize an empty list to store lemmatized words for the current row
        lemmatized_row = []
        # Concatenate all words from the current row into a single sentence
        sentence = ' '.join(row)
        # Process the sentence using spaCy
        doc = nlp(sentence)
        # Iterate over tokens and append lemmatized words to the list
        for token in doc:
            lemmatized_row.append(token.lemma_)
        # Append lemmatized words for the current row to the list of lemmatized words
        lemmatized_words.append(lemmatized_row)

    # Store lemmatized words back into the DataFrame
    df['Review'] = lemmatized_words


# In[51]:


lemmatize_dataframe(df)


# In[52]:


df.head()


# In[53]:


df['Processed_Review'] = df['Review'].apply(lambda x: ' '.join(x))  # x tokens concatenate to string with spaces in between


# In[54]:


# Initialize the vectorizer that convert the text documents into a matrix of token counts.
vectorizer = CountVectorizer()


# In[55]:


# fit_transform study the vocabulary from df['Processed_Review']
# transfer into matrix: each row = document, each column = word. value = number of appearences of the word in the document
X_bow = vectorizer.fit_transform(df['Processed_Review'])


# In[56]:


# Word Embedding with Word2Vec

# Create a list of token lists for Word2Vec training
token_lists = df['Review'].tolist()


# In[57]:


# vector_size = The dimensionality of the word vectors.
model_w2v = Word2Vec(sentences=token_lists, vector_size=100, window=5, min_count=1, workers=4)


# In[58]:


# Function calculate the average vector for document of the words 
def document_vector(word_list, model):
    # remove out-of-vocabulary words
    word_list = [word for word in word_list if word in model.wv.index_to_key]
    if len(word_list) == 0:
        # return vector from 0 if there is no correct word in the document
        return np.zeros(model.vector_size)
    else:
        # return the average vector of words that are in the  word_list
        return np.mean(model.wv[word_list], axis=0)


# In[59]:


# Apply the function to each row of the DataFrame
df['Document_Vector_filter'] = df['Review'].apply(lambda x: document_vector(x, model_w2v))


# In[60]:


# Convert the list of vectors into a 2D array
X_w2v = np.array(df['Document_Vector_filter'].tolist())


# In[61]:


# Split data into train and test sets
X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, df['Liked'], test_size=0.2, random_state=48)


# In[62]:


# Train SVM classifier with Bag-of-Words features
svm_bow = SVC(kernel='linear')
svm_bow.fit(X_train_bow, y_train)


# In[63]:


# Make predictions on test data
y_pred_bow = svm_bow.predict(X_test_bow)
accuracy_bow = accuracy_score(y_test, y_pred_bow)


# In[64]:


#accuracy for the Bag Word
print('\n')
print("Bag Of Words Accuracy: ", accuracy_bow * 100, "%")
print("Bag Of Words Classification Report: ")
print(classification_report(y_test, y_pred_bow))


# In[65]:


# Apply the SVM  on W2V "Word 2 vectors "

# Train Logistic Regression classifier with Word2Vec embeddings
logreg_w2v = LogisticRegression(max_iter=500)
logreg_w2v.fit(X_train_bow, y_train)

y_pred_logreg_w2v = logreg_w2v.predict(X_test_bow)
accuracy_logreg_w2v = accuracy_score(y_test, y_pred_logreg_w2v)


# In[66]:


print('\n')
print("Logistic Regression with Word2Vec Accuracy: ", accuracy_logreg_w2v * 100, "%")
print("Logistic Regression with Word2Vec Classification Report: ")
print(classification_report(y_test, y_pred_logreg_w2v))


# In[67]:


# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# In[68]:


def get_sentiment(text):

    scores = analyzer.polarity_scores(text)

    sentiment = 1 if scores['pos'] > 0 else 0

    return sentiment


# In[69]:


# apply get_sentiment function
df['scores'] = df['Processed_Review'].apply(analyzer.polarity_scores)
df['sentiment'] = df['Processed_Review'].apply(get_sentiment)
df['Liked'] = df['Liked'].map({'Yes': 1, 'No': 0})


# In[70]:


print('\n')
print("Confusion Matirx : ")
print(confusion_matrix(df['Liked'], df['sentiment']))


# In[71]:


print('\n')
print("Classification Report : ")
print(classification_report(df['Liked'], df['sentiment']))


# In[ ]:





# In[ ]:





# In[ ]:




