import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
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

