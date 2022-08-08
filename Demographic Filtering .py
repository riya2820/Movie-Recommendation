#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval


# In[2]:


path = '/Users/XXX/Downloads/XXX'
credits_df = pd.read_csv(path + "/tmdb_5000_credits.csv")
movies_df = pd.read_csv(path + "/tmdb_5000_movies.csv")


# In[3]:


movies_df.head()


# In[4]:


# Alogrithm 1 
# Demographic Filtering Algorithm 
C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.9)

print("90% percentile")
print("C:", C)
print("m:", m)
new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]
print("Shape =", new_movies_df.shape, "\n")

print("25% percentile")
m = movies_df["vote_count"].quantile(0.25)
print("C:", C)
print("m:", m)
new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]
print("Shape =", new_movies_df.shape, "\n")

print("75% percentile")
m = movies_df["vote_count"].quantile(0.75)
print("C:", C)
print("m:", m)
new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]
print("Shape =", new_movies_df.shape, "\n")


# In[5]:


def weighted_rating(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v/(v + m) * R) + (m/(v + m) * C)


# In[6]:


new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
new_movies_df = new_movies_df.sort_values('score', ascending=False)

print("*************** Top 10 ******************** \n")
new_movies_df[["title", "vote_count", "vote_average", "score"]].head(10)


# In[7]:


new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
new_movies_df = new_movies_df.sort_values('score', ascending=True)
print("*************** Bottom 10 ******************** \n")
new_movies_df[["title", "vote_count", "vote_average", "score"]].head(10)


# In[8]:


# Plot top 10 movies
def plot():
    popularity = movies_df.sort_values("popularity", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.barh(popularity["title"].head(10), popularity["popularity"].head(10), align="center", color="skyblue")
    plt.gca().invert_yaxis()
    plt.title("Top 10 movies")
    plt.xlabel("Popularity")
    plt.show()


# In[ ]:




