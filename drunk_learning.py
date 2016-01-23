import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier, PassiveAggressiveClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn import preprocessing

class DrunkLearningBatch(object):
    """drunk_learning class"""
    def __init__(self):
        self.clf = SVC()
        self.filename = 'modelSVM.pkl'

    def fit(self, X, y):
    	X = np.array([X])
        y = np.array(y)
        self.clf.fit(X, y)
        joblib.dump(self.clf, self.filename, compress=9)

    def predict(self, X):
    	X = np.array([X])
        ret = self.clf.predict(X)
        return str(ret[0])
    
    def grid_search(self, X, y, param_grid):
    	model = GridSearchCV(estimator=self.clf, param_grid=param_grid, cv=10)
	model.fit(X, y)

	print("Best parameters set found for Random Forest:")
	print("")
	print(model.best_estimator_)

	print("Grid scores:")
	print()
	for params, mean_score, scores in model.grid_scores_:
    	    print("%0.3f (+/-%0.03f) for %r"
            % (mean_score, scores.std() / 2, params))
	    print("")

class DrunkLearningAdaBoost(DrunkLearningBatch):
    """drunk_learning implementation of AdaBoost"""
    def __init__(self):
        super(DrunkLearningAdaBoost, self).__init__()
        self.clf = AdaBoostClassifier(base_estimator=DecisionTreeRegressor(max_depth=None), n_estimators=100)
        self.filename = 'modelAdaBoost.pkl'

class DrunkLearningRandomForest(DrunkLearningBatch):
    """drunk_learning implementation of a Random Forest Classifier"""
    def __init__(self):
        super(DrunkLearningRandomForest, self).__init__()
        self.clf = RandomForestClassifier(n_estimators = 250, max_features='sqrt', max_depth = None)
        self.filename = 'modelRandomForest.pkl'

class DrunkLearningOnline(DrunkLearningBatch):
    """drunk_learning class for online learning"""
    def __init__(self):
    	super(DrunkLearningOnline, self).__init__()
        self.clf = Perceptron()
        self.filename = 'modelPerceptron.pkl'

    def partial_fit(self, X, y):
        X = np.array([X])
        y = np.array(y)
        self.clf.partial_fit(X, y, [0, 1])
        joblib.dump(self.clf, self.filename, compress=9)

class DrunkLearningOnlineSVM(DrunkLearningOnline):
    """drunk_learning implementation of Online Linear SVM"""
    def __init__(self):
        super(DrunkLearningOnlineSVM, self).__init__()
        self.clf = SGDClassifier(loss="hinge", penalty="l2")
        self.filename = 'modelOnlineSVM.pkl'

class DrunkLearningNB(DrunkLearningOnline):
    """drunk_learning implementation of Online Naive Bayes"""
    def __init__(self):
        super(DrunkLearningNB, self).__init__()
        self.clf = GaussianNB()
        self.filename = 'modelNB.pkl'

class DrunkLearningPassiveAggressive(DrunkLearningOnline):
    """drunk_learning implementation of Online Passive Aggressive Classifier"""
    def __init__(self):
        super(DrunkLearningPassiveAggressive, self).__init__()
        self.clf = PassiveAggressiveClassifier(loss='hinge', C=1.0))
        self.filename = 'modelPassiveAggressive.pkl'

