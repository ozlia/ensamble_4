import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB


class ModelsAbstractClass:

    def __init__(self):
        adaboost = AdaBoostClassifier()
        gausianNB = GaussianNB()
        random_forest = RandomForestClassifier(random_state=2)
        lightgbm = LGBMClassifier(random_state=2)
        sdt = DecisionTreeClassifier(random_state=2)
        self.models = [random_forest, lightgbm, sdt, adaboost, gausianNB]
        self.train_x = []
        self.train_y = []

    def predict(self, test_x: pd.DataFrame):
        predictions = []
        for index, sample in test_x.iterrows():
            predictions.append(self.predict_one(sample))
        return predictions

    def export_trained_models_as_list(self):
        return self.models.copy()

    def import_models_from_list(self, model_list):
        self.models = model_list.copy()
