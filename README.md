# A Movies Recommendation System
In this repository is developed a simple recommendation engine for movies using collaborative filtering from lowrank matrix factorization inplemented in python. This project is based on data avaliable in the in this [movie review](http://www.cs.cornell.edu/people/pabo/movie-review-data/).

## Extracting Information from the Data Set.
Ratings of many movies given for four reviewers (users) are avaliable in the 

## Development guide 
Read this [guide](https://github.com/juandados/movies_recommendation/blob/master/movie_recomender_development_guide.ipynb) to get details about how this system was developed.

## How to execute it
This movie recommendation system is implemented in a python script. So a complete python 3 installation is necessary (however, an alternative using Docker is explained bellow). The following command executes the movies recommendation engine for a user identifyed as _Steve+Rhodes_, making 6 recommendations, and showing the top 8 historical ratings.
```bash
python recommender.py '{"user id":"Steve+Rhodes", "Recommendations limit": 5, "Historical limit":8}'
```
### Running it in Docker
If you don't have a full python installation (e.g. one with anaconda 3) run the next lines after install [Docker](https://docs.docker.com/install/).
```bash
sudo docker build -t movies_recomender .
sudo docker run -ti -v $(pwd)/:/tmp movies_recomender python recommender.py '{"user id":"Steve+Rhodes", "Recommendations limit": 5, "Historical limit":8}'
```

## References

1. [Generals on movies recommendation systems.](https://blog.statsbot.co/recommendation-system-algorithms-ba67f39ac9a3)
1. [Matrix factorization recommender.](https://beckernick.github.io/matrix-factorization-recommender/)
2. [A movie recommendation system inplemented on Spark.](https://www.packtpub.com/books/content/building-recommendation-engine-spark)
3. [About the Netflix recommendation system.](https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429)
4. [Performance metrics.](https://en.wikipedia.org/wiki/Information_retrieval#Precision_at_K)
5. [Movie ratings dataset.](https://grouplens.org/datasets/movielens/)
