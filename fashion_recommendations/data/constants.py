import pandas as pd


def gen_article_dict():
    articles_df = pd.read_csv(
        'data/articles.csv',
        dtype={'article_id': str}  # Make sure article_id is being loading in as a string
    )

    return dict(
        zip(
            articles_df['article_id'],
            articles_df.index + 2
        )
    )


ARTICLE_ID_TO_IDX = gen_article_dict()

TARGET_PROP_NO_HISTORY = 0.0070693450341841714

NO_HISTORY_ARTICLE_ID_IDX = '1'

RANDOM_SEED = 3
