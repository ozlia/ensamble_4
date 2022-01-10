from sklearn.cluster import KMeans
import pandas as pd
from models_abstract_class import ModelsAbstractClass


class ClusteringSelection(ModelsAbstractClass):
    def __init__(self):
        super(ClusteringSelection, self).__init__()
        self.train_clusters = {}
        self.best_models_per_cluster = {}
        self.kmeans = KMeans(random_state=2)

    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame , without_models=False):
        self.train_x = train_x.reset_index().drop(columns='index')
        self.train_y = train_y.reset_index().drop(columns='index')
        if not without_models:
            for m in self.models:
                m.fit(self.train_x, self.train_y)

        self.kmeans.fit(train_x)
        temp_models_score = {}
        for index, row in self.train_x.iterrows():
            cluster_num = self.kmeans.predict([row])[0]
            if cluster_num not in temp_models_score:
                temp_models_score[cluster_num] = {}
            for m in self.models:
                if m.predict([self.train_x.iloc[index]]) == self.train_y.iloc[index].values[0]:
                    temp_models_score[cluster_num][m] = temp_models_score[cluster_num][m] + 1 if m in temp_models_score[
                        cluster_num] else 1
        for cluster in temp_models_score.keys():
            self.best_models_per_cluster[cluster] = sorted(temp_models_score[cluster].items(), key=lambda x: x[1])[0][0]

    def predict_one(self, sample):
        cluster_for_sample = self.kmeans.predict([sample])[0]
        best_model = self.best_models_per_cluster[cluster_for_sample]
        return best_model.predict([sample])
