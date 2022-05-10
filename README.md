# fashion-recommendations

Had a go at the [H&M Personalized Fashion Recommendations Kaggle Competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) as a first venture into the world of recommendation systems.

Determined to use a Deep Learning solution rather than the popular `LGBMRanker`.

After ~20 notebooks with very poor public leaderboard performance I found that framing the problem as a multi-class, 
multi-label classification led to stable learning and reasonable leaderboard performance.

We are provided with numerous article (item) features but I found I was able to construct representative embeddings 
by simply applying a sentence encoder to the detailed description of each item. I took each customer's 100 most recent 
purchases and averaged over the item embeddings. Also trained embeddings for users, postcodes and categorical customer 
features from scratch.

Ended with a score of 0.01641 on the private leaderboard. Winning submission scored 0.03792, so room for improvement! Final training/submission notebook [here](https://github.com/stevieg3/fashion-recommendations/blob/master/notebooks/37-Multi-label%20model_more%20training%20data%20and%20postcode.ipynb) and data processing [here](https://github.com/stevieg3/fashion-recommendations/blob/master/notebooks/36-Data%20prep%20for%20multi-label_counts%20and%20age_more%20training%20data_more%20history%20and%20postcode_for_submission.ipynb).

If I had more time I would have:
1. Used my model as a candidate generator and trained a LGBM model on top of that
2. Created image embeddings using an AutoEncoder
3. Explored alternatives to averaging item embeddings such as LSTM or attention-based architectures





