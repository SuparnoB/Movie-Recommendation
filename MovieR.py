import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval

st.set_page_config(page_title="Movie Recommendations", page_icon="📈")
st.markdown("<h1 style='text-align: center; color: lightblue;'>Movie Recommendation System</h1>", unsafe_allow_html=True)

d1=pd.read_csv('C:/Users/Suparno Victus/Movie_Recomm/imdb_5000_credits.csv')
d2=pd.read_csv('C:/Users/Suparno Victus/Movie_Recomm/imdb_5000_movies.csv')

d1.columns = ['id','title','cast','crew']
d2 = d2.merge(d1,on='id')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

def check_df(dataframe, head=5, tail=5, quan=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### Sum of Missing #####################")
    print(dataframe.isnull().sum())
    print("##################### Percantage of Missing #####################")
    print(100 * dataframe.isnull().sum() / len(dataframe))

    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
d2['overview'] = d2['overview'].fillna('')

selectedOption = st.selectbox("Select the Movie watched to get 10 reccomendation!", d2['original_title'].tolist())
productName = d2[d2['original_title'] == selectedOption].head(1).iloc[0]['overview']
st.write(productName)

    #Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(d2['overview'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(d2.index, index=d2['original_title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return d2['original_title'].iloc[movie_indices]


def findProductNames(get_recommendations):
    names = []
    st.info("RESULTS")
    for item in get_recommendations:
        df_row = d2[d2['original_title'] == item].head(1)
        names.append(item +" - "+ df_row.iloc[0]['overview'])
        st.header(item)
        st.write(df_row.iloc[0]['overview'])
a = get_recommendations(selectedOption, cosine_sim=cosine_sim)
findProductNames(a)