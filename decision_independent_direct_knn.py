from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from decision_dependent_direct_knn import DecisionDependentDirectKNN


class DecisionIndependentDirectKNN(DecisionDependentDirectKNN):
    def predict_one(self, sample):
        nn = self.knn.kneighbors([sample], 5, return_distance=False)
        model_pred_score = {}
        for n in nn[0]:
            for m in self.models:
                if m.predict([self.train_x[n]]) == self.train_y[n]:
                    model_pred_score[m] = model_pred_score[m] + 1 if m in model_pred_score else 1
        best_model = [(k, v) for k, v in sorted(model_pred_score.items(), key=lambda item: item[1], reverse=True)][0][0]
        return best_model.predict([sample])[0]
