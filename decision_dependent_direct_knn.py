import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier


class DecisionDependentDirectKNN:

    def __init__(self):
        # todo choose 2 more models as we wish
        self.knn = KNeighborsClassifier()
        random_forest = RandomForestClassifier()
        lightgbm = LGBMClassifier()
        sdt = DecisionTreeClassifier()
        self.models = [random_forest, lightgbm, sdt]
        self.train_x = []
        self.train_y = []

    def fit(self, train_x, train_y):
        for m in self.models:
            m.fit(train_x, train_y)
        self.knn.fit(train_x, train_y)
        self.train_x = train_x
        self.train_y = train_y

    def predict_one(self, sample):
        nn = self.knn.kneighbors([sample], 5, return_distance=False)
        model_pred_score = {}
        for n in nn[0]:
            for m in self.models:
                if m.predict([self.train_x[n]]) == self.train_y[n] == m.predict([sample]):
                    model_pred_score[m] = model_pred_score[m] + 1 if m in model_pred_score else 1
        best_model = [(k, v) for k, v in sorted(model_pred_score.items(), key=lambda item: item[1], reverse=True)][0][0]
        return best_model.predict([sample])[0]

    def predict(self, test_x:pd.DataFrame):
        predictions = []
        for index, sample in test_x.iterrows():
            predictions.append(self.predict_one(sample))
        return predictions
