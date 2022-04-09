import pandas as pd

from fashion_recommendations.data.constants import (
    TARGET_PROP_NO_HISTORY,
    NO_HISTORY_ARTICLE_ID_IDX,
    RANDOM_SEED
)


def prepare_splits(train_df, dev_df, name):

    # ================================================================================================================ #
    # Prepare training data
    # ================================================================================================================ #
    train_df = train_df.copy()

    train_df.reset_index(drop=True, inplace=True)
    # Create a UID as a customer may purchase the same item multiple times preventing a 1:1 join:
    train_df['uid'] = train_df.index

    # Sample transactions by customer to use as labels
    # labels = train_df[['customer_id', 'uid']].groupby('customer_id').sample(n=1, random_state=RANDOM_SEED)
    labels = train_df[['customer_id', 'uid']].groupby('customer_id').tail(1)  # Use last transaction as label
    labels['label'] = 1

    # Drop customer transactions after the label
    train_df = train_df.merge(labels, on=['customer_id', 'uid'], how='left')
    train_df['keep'] = train_df['label'].copy()
    train_df['keep'] = train_df.groupby('customer_id')['keep'].bfill()
    train_df = train_df.copy()[train_df['keep'] == 1]
    train_df['label'].fillna(0, inplace=True)
    train_df.drop(columns=['uid', 'keep'], inplace=True)

    # Collapse on customer and concatenate article_id_idx into a single string
    train_df['article_id_idx'] = train_df['article_id_idx'].astype(str)

    train_historical_purchases = (
        train_df[train_df['label'] == 0][['customer_id', 'article_id_idx']]
        .groupby('customer_id')
        .agg({'article_id_idx': ','.join})
        .reset_index()
    )

    train_labels = train_df.copy()[train_df['label'] == 1][['customer_id', 'article_id_idx']]

    # Combine historical transaction strings and labels:
    train_combined = train_labels.merge(
        train_historical_purchases,
        on='customer_id',
        how='left',
        suffixes=('_label', '_historical'),
        indicator=True
    )

    # Downsample customers with no transaction history

    number_customers_both = train_combined[train_combined['_merge'] == 'both'].shape[0]

    number_to_sample = ((number_customers_both / (1-TARGET_PROP_NO_HISTORY)) - number_customers_both)
    number_to_sample = int(number_to_sample)

    train_combined_new = pd.concat(
        [
            train_combined[train_combined['_merge'] == 'both'],
            train_combined[train_combined['_merge'] != 'both'].sample(n=number_to_sample, random_state=RANDOM_SEED)
        ]
    )

    prop = train_combined_new[train_combined_new['_merge'] == 'left_only'].shape[0] / train_combined_new.shape[0]
    print(f"Train no history customers prop: {prop}")

    train_combined_new.drop('_merge', axis=1, inplace=True)

    # For customers with no history impute fixed ID
    train_combined_new['article_id_idx_historical'].fillna(NO_HISTORY_ARTICLE_ID_IDX, inplace=True)

    # Shuffle dataset (for IterableDataset)
    train_combined_new = train_combined_new.sample(frac=1, random_state=RANDOM_SEED)

    # Save
    train_combined_new.to_csv(f'data/splits/train_single_purchase_label_{name}.tsv', sep='\t', index=False)

    if dev_df is None:
        print("No dev data, ending")
        return None

    # ================================================================================================================ #
    # Prepare dev data
    # ================================================================================================================ #

    dev_df = dev_df.copy()

    dev_df['article_id_idx'] = dev_df['article_id_idx'].astype(str)

    # Create all purchase label (for MAP@k)
    dev_labels_all_purchases = (
        dev_df[['customer_id', 'article_id_idx']]
        .groupby('customer_id')
        .agg({
            'article_id_idx': ','.join
        })
        .reset_index()
    )

    # Create single purchase label (for loss)
    dev_labels_single_label = dev_df.copy()[['customer_id', 'article_id_idx']].groupby('customer_id').head(1)

    assert dev_labels_all_purchases.shape[0] == dev_labels_single_label.shape[0]

    # Merge labels:
    dev_labels = dev_labels_all_purchases.merge(
        dev_labels_single_label, on='customer_id', suffixes=('_all_purchases', '_single_label')
    )

    # Merge on historical purchases from training data
    dev_labels = dev_labels.merge(
        train_combined_new, on='customer_id', how='left', indicator=True
    )

    # Downsample no history customers
    number_customers_both = dev_labels[dev_labels['_merge'] == 'both'].shape[0]

    number_to_sample = ((number_customers_both / (1-TARGET_PROP_NO_HISTORY)) - number_customers_both)
    number_to_sample = int(number_to_sample)

    dev_labels_new = pd.concat(
        [
            dev_labels[dev_labels['_merge'] == 'both'],
            dev_labels[dev_labels['_merge'] != 'both'].sample(n=number_to_sample, random_state=RANDOM_SEED)
        ]
    )

    prop = dev_labels_new[dev_labels_new['_merge'] == 'left_only'].shape[0] / dev_labels_new.shape[0]
    print(f"Dev no history customers prop: {prop}")

    dev_labels_new.drop('_merge', axis=1, inplace=True)

    dev_labels = dev_labels_new.copy()

    dev_labels['article_id_idx_historical'].fillna(NO_HISTORY_ARTICLE_ID_IDX, inplace=True)

    # All purchase label
    (
        dev_labels[['customer_id', 'article_id_idx_all_purchases', 'article_id_idx_historical']]
        .rename(columns={'article_id_idx_all_purchases': 'article_id_idx_label'})
        .to_csv(f'data/splits/dev_all_purchase_label_{name}.tsv', sep='\t', index=False)
    )

    # Single purchase label
    (
        dev_labels[['customer_id', 'article_id_idx_single_label', 'article_id_idx_historical']]
        .rename(columns={'article_id_idx_single_label': 'article_id_idx_label'})
        .to_csv(f'data/splits/dev_single_purchase_label_{name}.tsv', sep='\t', index=False)
    )
