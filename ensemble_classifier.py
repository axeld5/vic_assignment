import numpy as np
import time
import pickle

from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

class EnsembleClassifier:

    def __init__(self, use_svm, use_rf, use_xgb) -> None:
        n_classifiers = int(use_svm) + int(use_rf) + int(use_xgb)
        self.weights = [1/n_classifiers for i in range(n_classifiers)]
        self.clf_list = []
        if use_svm:
            self.clf_list.append(SVC(probability=True)) 
        if use_rf:
            self.clf_list.append(RandomForestClassifier())
        if use_xgb:
            self.clf_list.append(HistGradientBoostingClassifier())

    def fit(self, x_train, y_train):
        for model in self.clf_list:
            start = time.time()
            model.fit(x_train, y_train) 
            print(time.time() - start)

    def predict_proba(self, x):
        proba_arr = np.zeros((len(x), 2))
        for i, model in enumerate(self.clf_list):
            predicted_proba = model.predict_proba(x)
            proba_arr[:,0] += predicted_proba[:,0]*self.weights[i]
            proba_arr[:,1] += predicted_proba[:,1]*self.weights[i]
        return proba_arr

    def predict(self, x):
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)
    
    def save_models(self):
        for i, model in enumerate(self.clf_list):
            pkl_filename = "pickle_model_"+str(i)+".pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(model, file)

    def load_models(self):
        for i, model in enumerate(self.clf_list):
            pkl_filename = "pickle_model_"+str(i)+".pkl"
            with open(pkl_filename, 'rb') as file:
                pickle_model = pickle.load(file)
            self.clf_list[i] = pickle_model