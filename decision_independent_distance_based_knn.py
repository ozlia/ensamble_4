from decision_dependent_direct_knn import DecisionDependentDirectKNN


class DecisionIndependentDistanceBasedKNN(DecisionDependentDirectKNN):
    def predict_one(self, sample):
        nn = self.knn.kneighbors([sample], 5, return_distance=False)
        model_pred_score = {}
        i = 0
        for n in nn[0]:
            i += 1
            for m in self.models:
                if m.predict([self.train_x[n]]) == self.train_y[n] == m.predict([sample]):
                    rank = 1/i
                    model_pred_score[m] = model_pred_score[m] + rank if m in model_pred_score else rank
        best_model = [(k, v) for k, v in sorted(model_pred_score.items(), key=lambda item: item[1], reverse=True)][0][0]
        return best_model.predict([sample])[0]
