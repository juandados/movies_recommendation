import sys
import json
import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from scipy.sparse.linalg import svds

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
            words_array = re.findall('([A-Z][A-Z\:\-\!\/\'\s\d]+) [a-z]',review_text)
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
    def fit(cls):
        '''
        It determines the matrix descomposition (sigular value matrix descomposition), which will be use to stimate ratings
        '''
        R_df = cls.ratings_df.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        cls.users_index = R_df.index.values
        cls.movies_index = R_df.columns.values
        cls.U, cls.sigma, cls.Vt = svds(R_df.as_matrix(),k=np.min([(np.min(R_df.shape)-1),8]))

    def __init__(self, userId):
        self.userId = userId


    def recommend(self,limit=4):
        '''
        It makes a specified number of recommendations, based on higher estimated ratings
        '''
        cls = self.__class__
        seen_movies = cls.ratings_df.loc[cls.ratings_df["userId"] == self.userId]["movieId"]
        estimated_ratings = self.estimate_ratings()
        not_seen_estimated_ratings = estimated_ratings.loc[estimated_ratings["movieId"].map(lambda x: not x in seen_movies.values)]
        top_recommendations = not_seen_estimated_ratings.sort_values("estimated_rating",ascending=False).head(limit)
        top_recommendations = pd.merge(top_recommendations,cls.movies_df, on = "movieId")
        del top_recommendations["estimated_rating"]
        return top_recommendations

    def estimate_ratings(self):
        '''
        It estimated the rating for every movie in the original rating list, based on the SVD factorization
        '''
        cls = self.__class__
        estimated_ratings = np.dot(cls.U[self.userId-1],np.dot(np.diag(cls.sigma),cls.Vt))
        estimated_ratings_df = pd.DataFrame([cls.movies_index,estimated_ratings],index=["movieId","estimated_rating"]).T
        return estimated_ratings_df

    def get_top_historical_ratings(self, limit=10):
        '''
        It get the limit-top historical rating for a specific user
        '''
        cls = self.__class__
        ratings = cls.ratings_df.loc[cls.ratings_df["userId"] == self.userId]
        top_historical_ratings = ratings.sort_values("rating",ascending = False)
        top_historical_ratings = pd.merge(top_historical_ratings.head(limit),cls.movies_df, on="movieId")
        return top_historical_ratings


if __name__ == '__main__':
    try:
        names = ["Dennis+Schwartz","James+Berardinelli","Scott+Renshaw","Steve+Rhodes"]
        for name in names:
            data_loader = DataLoader(name)
            data_loader.filling_reviews_titles()
        movies_df = pd.DataFrame([DataLoader.review_titles],index=["movieId"]).T
        movies_df.reset_index(inplace=True)
        movies_df.rename(columns={"index":"reviewId"},inplace=True) 
        movies_df["title"] = movies_df["movieId"]
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
        ratings_df = ratings_df.groupby(["userId","movieId"]).head(1)

        inputs = json.loads(sys.argv[1])
        # Load the data to the MovieRecommender class
        MovieRecommender.load_data_set(ratings_df = ratings_df, movies_df = movies_df)
        # Fit the recommendation engine 
        MovieRecommender.fit()
        # Make an intance of the recommender
        recommender = MovieRecommender(userId = inputs["user id"])
        # Ask a list of recommendations
        print("-----------------------")
        print("Our recommendations are")
        print("-----------------------")
        print(recommender.recommend(limit = inputs.setdefault("Recommendations limit",4)).to_string())
        # To compare the above result call for the top historical of ratings for the given user
        print("----------------------")
        print("Because you watched...")
        print("----------------------")
        print(recommender.get_top_historical_ratings(limit = inputs.setdefault("Historicals limit",10)).to_string())
    except:
        print("Something went wrong!")
