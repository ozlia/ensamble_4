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
        """

        :param test_x: test data as df
        :return: list of y_pred
        """
        predictions = []
        for index, sample in test_x.iterrows():
            predictions.append(self.predict_one(sample))
        return predictions

    def export_trained_models_as_list(self):
        """

        :return:export list of pre trained model
        """
        return self.models.copy()

    def import_models_from_list(self, model_list):
        """

        :param model_list: model list of pretrained models
        :return:
        """
        self.models = model_list.copy()

    def predict_one(self, sample):
        """

        :param sample: sample to predict
        :return: predicted value of sample by chosen model
        """
        pass

    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame, without_models=False):
        """

        :param train_x: x train data as DF
        :param train_y: x train data as DF
        :param without_models: if true, its mean that pretrained model is already imported, and there is no necessary for train all model again
        :return: None
        """
        pass
