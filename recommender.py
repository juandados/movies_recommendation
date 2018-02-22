import sys
import json
import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from scipy.sparse.linalg import svds
from numpy.linalg import norm

class DataLoader:
    base_dir = "./cornell-data/scale_whole_review/scale_whole_review/{}/txt.parag"
    review_titles = {}
    
    def __init__(self, name):
        self.name = name
        if name == "Dennis+Schwartz":
            self.title_setter = self.title_setter_by_director_token
        else:
            self.title_setter = self.title_setter_by_frequence
    
    def filling_reviews_titles(self):
        directory = self.base_dir.format(self.name)
        #Selecting an title setter:
        review_files = os.listdir(directory)
        for review_file in review_files:
            review_text = self.open_file(directory,review_file)
            review_id = int(re.split("\.",review_file)[0])
            s = self.title_setter(review_text, review_id)
            
    def open_file(self,directory,review_file):
        with open("{}/{}".format(directory,review_file), "rb") as file:
            text_review = file.read().decode('utf-8',errors='ignore')
        return text_review
    
    def title_setter_by_director_token(self,review_text, review_id): 
        try:
            title = re.findall('([A-Z,.\:\-\!\/\'\s\d]+)\s+.*\([dD]irect', review_text)
            self.review_titles[review_id] = re.split("\,",title[0])[0]
        except:
            pass

    def title_setter_by_frequence(self,review_text, review_id):
        try:
            words_array = re.findall('([A-Z][A-Z\:\-\!\/\'\s\d\&]+) [a-z]',review_text)
            title = Counter(words_array).most_common(1)[0][0]
            self.review_titles[review_id] = title
        except:
            pass

class MovieRecommender:

    @classmethod
    def load_data_set(cls, ratings_df, movies_df):
        '''
        It defines two class attributes linked with the data set
        '''
        cls.ratings_df = ratings_df
        cls.movies_df = movies_df

    @classmethod
    def matrix_decomposition(cls):
        '''
        It determines the matrix descomposition (sigular value matrix descomposition), which will be use to stimate ratings
        '''
        cls.R_df = cls.ratings_df.pivot(index="userId", columns="movieId", values="rating").fillna(-1)
        cls.users_index = cls.R_df.index.values
        cls.movies_index = cls.R_df.columns.values
        cls.U, cls.sigma, cls.Vt = svds(cls.R_df.as_matrix(),k=np.min([(np.min(cls.R_df.shape)-1),8]))

    def __init__(self, userId):
        self.userId = userId


    def recommend(self,recommendations_limit=4, historical_limit=10):
        '''
        It makes a specified number of recommendations, based on higher estimated ratings
        '''
        ratings = self.R_df.loc[self.userId]
        self.best_ranked_movies = ratings[ratings>0.7].sort_values(ascending=False).head(historical_limit)
        ## Selecting Features for unseen movies
        Vt_df = pd.DataFrame(self.Vt,columns=self.R_df.columns)
        unseen_movies = Vt_df[self.R_df.columns[ratings == -1]]
        ### Calculating Distances over Vt Matrix
        differences = []
        for movie in Vt_df[self.best_ranked_movies.index].items():
            differences.append(unseen_movies.apply(lambda x: norm(x-movie[1])).values)
        distances = pd.DataFrame(differences,columns=unseen_movies.columns)
        ds = list(set(distances.values.flatten())).sort()
        indexes = distances.apply(lambda x: x.apply(lambda y: y in ds[:recommendations_limit]))
        movie_lowest_score = distances[indexes].min()
        top_recommendations = movie_lowest_score[~movie_lowest_score.isnull()].sort_values().head(recommendations_limit)
        top_recommendations = pd.Series(top_recommendations.index)
        return top_recommendations 

    def get_top_historical_ratings(self):
        return self.best_ranked_movies

if __name__ == '__main__':
    try:
        names = ["Dennis+Schwartz","James+Berardinelli","Scott+Renshaw","Steve+Rhodes"]
        # Extracting titles from reviews
        for name in names:
            data_loader = DataLoader(name)
            data_loader.filling_reviews_titles()
        ## Creating movies DataFrame
        movies_df = pd.DataFrame([DataLoader.review_titles],index=["movieId"]).T
        movies_df.reset_index(inplace=True)
        movies_df.rename(columns={"index":"reviewId"},inplace=True) 
        movies_df["title"] = movies_df["movieId"]
        ## Creating a ratings DataFrame
        id_base_dir = "./cornell-data/scale_data/scaledata/{}/id.{}"
        rating_base_dir = "./cornell-data/scale_data/scaledata/{}/rating.{}"
        ratings = pd.DataFrame([],columns=["reviewId","rating","userId"])
        for name in names:
            rates = pd.read_table(rating_base_dir.format(name,name),names=["rating"])
            ids = pd.read_table(id_base_dir.format(name,name),names=["reviewId"])
            user_ratings = pd.concat([ids,rates],axis=1)
            user_ratings["userId"] = name
            ratings = pd.concat([ratings,user_ratings])
        ratings_df = pd.merge(ratings,movies_df, on="reviewId")
        ### Droping duplicated reviews for a single user
        ratings_df = ratings_df.groupby(["userId","movieId"]).head(1)
        inputs = json.loads(sys.argv[1])
        # Load the data to the MovieRecommender class
        MovieRecommender.load_data_set(ratings_df = ratings_df, movies_df = movies_df)
        # Making Lower Rank Decomposition 
        MovieRecommender.matrix_decomposition()
        # Make an intance of the recommender
        recommender = MovieRecommender(userId = inputs["user id"])
        # Ask a list of recommendations
        print("-----------------------")
        print("Our recommendations are")
        print("-----------------------")
        print(recommender.recommend(recommendations_limit =  inputs.setdefault("Recommendations limit",4), historical_limit = inputs.setdefault("Historical limit",10)))
        # To compare the above result call for the top historical of ratings for the given user
        print("----------------------")
        print("Because you watched...")
        print("----------------------")
        print(recommender.get_top_historical_ratings().to_string())
    except:
        print("Something went wrong!")
