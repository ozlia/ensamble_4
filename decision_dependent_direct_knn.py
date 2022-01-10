import random
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from models_abstract_class import ModelsAbstractClass


class DecisionDependentDirectKNN(ModelsAbstractClass):

    def __init__(self):
        super(DecisionDependentDirectKNN, self).__init__()
        self.knn = KNeighborsClassifier()
        self.already_predicted = {}

    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame, without_models=False):
        if not without_models:
            for m in self.models:
                m.fit(train_x, train_y)
        self.train_x = train_x.reset_index().drop(columns='index')
        self.train_y = train_y.reset_index().drop(columns='index')
        self.knn.fit(self.train_x, self.train_y)

    def get_predict_from_ds(self, model, sample_n):
        """

        :param model: model to predict with
        :param sample_n: number of sample want to predict
        :return: fill the dict contain all predict group by model - the target is to use memory for fasten process
        """
        if model in self.already_predicted:
            if sample_n not in self.already_predicted[model]:
                self.already_predicted[model][sample_n] = model.predict([self.train_x.iloc[sample_n]])
        else:
            self.already_predicted[model] = {sample_n: model.predict([self.train_x.iloc[sample_n]])}

        return self.already_predicted[model][sample_n]

    def predict_one(self, sample):
        nn = self.knn.kneighbors([sample], 5, return_distance=False)
        model_pred_score = {}
        for n in nn[0]:
            for m in self.models:
                if self.get_predict_from_ds(m, n) == self.train_y.iloc[n].values[0] == m.predict([sample]):
                    model_pred_score[m] = model_pred_score[m] + 1 if m in model_pred_score else 1
        if len(model_pred_score) == 0:
            best_model = random.sample(self.models, 1)[0]
        else:
            best_model = \
                [(k, v) for k, v in sorted(model_pred_score.items(), key=lambda item: item[1], reverse=True)][0][0]
        return best_model.predict([sample])[0]
