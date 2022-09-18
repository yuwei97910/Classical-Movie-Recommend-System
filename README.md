# STAT542 Project4: Movie Recommendation System
### Here is my demo app on Herokuapp:
https://stat542-project4-recommend-sys.herokuapp.com

- For the model building process, please refer to the notebook: project4_summary
----
## Introduction

The user-based recommendation models are created by the package: Surprise, and the web app of the two systems is created by the package: Dash.
The testing area of these models is on MacBook Air(2021) with M1 chip, 8 cores CPU, 3.2 GHz, with 8 GB RAMs
The app was deployed with Github and hosted on Heroku Server.

----
## System 1: The determinant recommendations
Given the users' favorite genres and make recommendations for the top classic movies and trendy movies

For the System one, we are required to provide two sets of recommendations.
I try to recommend those movies which are more "classic" and "trending" movies for each genre.
Therefore, I first defined the two kinds of scores for each movie by the rating sets and the movies set.

### 1. The classic movies
The must see ones. High ranked by most reviewers & People always talked a lot about them

This is measured by overall rating numbers and the median of the rating score. The reason of using median is that it indicated that more than 50% of people will at least agree on this rating, which I considered to be more useful than the average score.

#### classic_score = rating count of a movie * average rating score
- Calculate the count of rating and normalized
- Calculate the median of rating
- Sum the two scores as 'classic_score'

### 2. The trending movies
People nowadays are talking about.
* there are 86400 seconds for one day
Followed the EDA of the dataset by Prof.Liang, the rating count will have huge impact on our recommendation. Threfore, to reduce those ratings that are relevantly old, I choose to remove the effect in this section.

#### (1) calculate the trending value for each rating:
- Calculate the time difference to present for each review
- Calculate the weighted rating scores as trending value for up to date ratings (the time difference less than mean minus one std)

#### (2) calculate the trending score for each movie:
- Sum all of the trending value for each movies as 'trending_score'


----
## System 2: The Recommender based on users' preferences
 
### Method 1: K-Nearest Neighbors
I first use the train-test set we created to fit a single model.
The model I used is the KNNBasic() model from Surprise. The algorithm aimed to optimized:

$$
\hat{r}_{ui} = \frac{\sum\limits_{v \in N^k_i(u)} \text{sim}(u, v) \cdot r_{vi}}{\sum\limits_{v \in N^k_i(u)} \text{sim}(u, v)}
$$

##### Parameters:
- K: The default setting is 40, which might be a large value. I choose to try k=10 at this step.
- Similarity measure:
 - User-based or Item-based: The default setting is User-based model; that is, the similarities (distance) are measured by users' properties
     * The similarities are computed between users
     * which is more reasonable to use on this system, since we can know the new users' rating preference when we make recommendations for them. Also, we had only the records of rating that could be used for measurement.
 - The algorithms measured similarities/distance by Mean Squared Difference (MSD) similarity: it only takes common users into consideration.
     * It compared the user distance on features, which is measured as:
     
 $$
 \text{msd}(u, v) = \frac{1}{|I_{uv}|} \cdot\sum\limits_{i \in I_{uv}} (r_{ui} - r_{vi})^2
 $$
 
(Surprise' Documentation 2021)

##### Result of the single model:
The approximate processing time is: 12.7 secs (it will take longer for making prediction)
The Rooted Mean Square Error (RMSE) for the testing set is 0.9272

-----
### Method 2: Singular Value Decomposotion
I first use the train-test set we created to fit a single model.

The model I used is the SVD() model from Surprise. In the algorithms, the predictor is:

$$
\hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u; \hat{r}_{ui} = q_i^Tp_u
$$

The algorithm aimed to minimize:

$$
\sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \right)^2 +\lambda\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\right)
$$

##### Parameters:
- Learning rate: the default of algorithms is 0.005
- Regularization term: the default of algorithms is 0.02
- The number of factors is set to be 100 and the number of iteration is set to be 20
(Surprise' Documentation 2021)

##### Result of the single model:
The approximate processing time is: 36.7 secs
The Rooted Mean Square Error (RMSE) for the testing set is 0.88

----
## Reference
- STAT542 Project 4 Guidelines on Campuswire
- Stackoverflows
- Surprise' Documentation. Welcome to Surprise' documentation! - Surprise 1 documentation. (2021). Retrieved December 12, 2021, from https://surprise.readthedocs.io/en/stable/index.html.
- Dash documentation &amp; user guide. Plotly. (2021). Retrieved December 12, 2021, from https://dash.plotly.com/.
- Lasseter, A. (2021, April 28). Deploy a plotly dash app on Heroku. Medium. Retrieved December 13, 2021, from https://austinlasseter.medium.com/deploy-a-plotly-dash-app-on-heroku-4d2c3224230.
