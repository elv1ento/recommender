import os
import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
import turicreate as tc
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing_tools import text_cleaner


class HybridRecommender:

    def __init__(self):
        self.search_engine = SearchEngine()
        self.collab_engine = CollaborativeEngine()
        self.content_engine = ContentEngine()
        # get_datasets
        self.books_df = pd.read_csv('data/books_cleaned.csv', low_memory=False).set_index('ISBN')
        self.fulltext_df = pd.read_csv('data/normalized_dask.csv').set_index('ISBN')

    def search(self, user_query, items_count=10):
        ranking_items = self.search_engine.get_results(user_query, items_count=items_count)
        return self.books_df.loc[ranking_items[:items_count]].iloc[:, :-3]

    def total_similarities(self, target_items):
        collab_items = self.collab_engine.get_results(target_items)

        target_nums = self.fulltext_df.loc[target_items]['item_id'].tolist()
        content_items = self.content_engine.get_results(target_nums)

        total = collab_items.join(content_items, how='outer').fillna(0)
        total = total.sum(axis=1).sort_values(ascending=False)[:20]

        return self.books_df[self.books_df.index.isin(total.index)]


class ModelManager:
    models_dir = 'models'
    data_dir = 'data'

    def __init__(self, model_name, dataset_name, is_exist=True):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model_path = os.path.join(self.models_dir, model_name)
        self.dataset_path = os.path.join(self.data_dir, dataset_name)
        self.dataset = pd.DataFrame()
        self.model = None
        self.load_dataset()

        if is_exist:
            self.load_model()

    def load_dataset(self):
        self.dataset = pd.read_csv(self.dataset_path)
        print(f'{self.dataset_name}: Dataset loaded.')

    def save_model(self):

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        pickle.dump(self.model, open(self.model_path, 'wb'))
        print(f'{self.model_name}: Model saved.')

    def load_model(self):
        self.model = pickle.load(open(self.model_path, 'rb'))
        print(f'{self.model_name}: Model loaded.')


# SEARCH

class SearchEngine(ModelManager):

    def __init__(self):
        super().__init__('search_model.pkl', 'normalized_dask.csv')

    def create_model(self):
        self.model = BM25Okapi(self.dataset['Fulltext'].apply(lambda x: x.split(' ')))
        print('Search model creared.')

    # Get results for a search query
    def get_results(self, user_query, items_count=10):
        tokenized_query = text_cleaner(user_query)
        doc_scores = self.model.get_scores(tokenized_query)
        score_dict = dict(zip(self.dataset.ISBN.values, doc_scores))
        doc_ranking = sorted(score_dict, key=score_dict.get, reverse=True)
        return doc_ranking[:items_count]


# Collaborative

class CollaborativeEngine(ModelManager):

    def __init__(self, data_load=False):
        self.data_load = data_load
        super().__init__('collaborative.model', 'book_ratings_cleaned.csv')

    def load_dataset(self):
        if self.data_load:
            self.dataset = tc.SFrame.read_csv(self.dataset_path)

    def create_model(self, user_col='user_id', item_col='ISBN'):
        train_data, valid_data = tc.recommender.util.random_split_by_user(
            self.dataset, user_col, item_col)

        self.model = tc.recommender.create(train_data, user_col, item_col)

    def save_model(self):
        self.model.save(self.model_path)

    def load_model(self):
        self.model = tc.load_model(self.model_path)

    def get_results(self, target_items, k=100):
        similar_items = self.model.get_similar_items(target_items, k=k)
        similar_df = similar_items.to_dataframe().drop(['ISBN'], axis=1)
        similar_df = similar_df.rename(columns={'similar': 'ISBN'})
        return similar_df.set_index('ISBN')[['score']]


# CONTENT

class ContentEmbedding(ModelManager):

    def __init__(self):
        super().__init__('tfidf_matrix.pkl', 'normalized_dask.csv')

    def create_model(self):
        self.model = TfidfVectorizer()
        self.model.fit_transform(self.dataset['Fulltext'].values)
        print('Search model created.')


class ContentEngine(ModelManager):

    def __init__(self):
        super().__init__('NN_cosine.pkl', '')

    def load_dataset(self):
        self.embedding_model = ContentEmbedding()
        print('Dataset loaded.')

    def create_model(self):
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=3, n_jobs=-1)
        self.model.fit(self.embedding_model)
        print('Search model creared.')

    # TODO: write comments
    def get_results(self, target_nums, n_neighbors=20):
        distances, indices = self.model.kneighbors(
            self.embedding_model.model[target_nums],
            n_neighbors=n_neighbors
        )

        similar_df = pd.DataFrame(
            zip(indices.ravel(), distances.ravel()),
            columns=['item_id', 'distance']
        )

        similar_df = similar_df.groupby('item_id').sum().sort_values(by='distance').reset_index()

        similar_df = pd.merge(
            similar_df,
            self.embedding_model.dataset.reset_index(),
            on='item_id')

        similar_df['sim_score'] = abs(
            MinMaxScaler().fit_transform(similar_df.distance.values.reshape(-1, 1)).ravel() - 1)

        return similar_df.set_index('ISBN')[['sim_score']]
