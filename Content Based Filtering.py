#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import sys 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval


# In[13]:


path = '/Users/XXX/Downloads/XXX'
credits_df = pd.read_csv(path + "/tmdb_5000_credits.csv")
movies_df = pd.read_csv(path + "/tmdb_5000_movies.csv")


# In[14]:


movies_df.head()


# In[ ]:


# Algorithm 2: Content Based Filtering


# In[41]:


# tf-idf scores 
tfidf = TfidfVectorizer(stop_words="english")
movies_df["overview"] = movies_df["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(movies_df["overview"])
print(tfidf_matrix.shape)


# In[43]:


# Computing cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print("Shape =", cosine_sim.shape, "\n")

indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()
print(indices.head())


# In[44]:


def get_recommendations(title, cosine_sim=cosine_sim):
    # sorting using cosine scored and then mapping those indices to their titles 
 
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # desceding order
    sim_scores = sim_scores[1:11]
    # (x, y) 
    # x = id of movie
    # y = sim_score

    movies_indices = [ind[0] for ind in sim_scores]
    movies = movies_df["title"].iloc[movies_indices]
    return movies


# In[55]:


print("---------- Content Based Filtering --------\n")
print("Movies you may like")
print("Since you watched Interstellar \n")
print(get_recommendations("Interstellar"))
print("\n")

print("Since you watched Avengers \n")
print(get_recommendations("The Avengers"))
print("\n")

print("Since you watched The Matrix \n")
print(get_recommendations("The Matrix"))
print("\n")


# In[ ]:




