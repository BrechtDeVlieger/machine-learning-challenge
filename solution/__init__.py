import re

import pandas as pd
import numpy as np
from sklearn_pandas import CategoricalImputer
try:
    import xgboost as xgb
except ImportError:
    xgb = None
from keras.constraints import max_norm
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelBinarizer, Imputer


class Drop(TransformerMixin):
    """ Pipeline transformer that drops columns from the dataframe """
    def __init__(self, columns, **kwargs):
        """ Drop columns from the dataframe 
        
        :param columns: List of column names to drop from the dataframe
        :param kwargs: Keyword arguments are passed to the DataFrame.drop method
        """
        self.cols = columns
        self.kwargs = kwargs
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.drop(columns=self.cols, **self.kwargs)
    

class MultiCategoricalImputer(TransformerMixin):
    """ Pipeline transformer to impute categorical features of type string """
    def __init__(self, columns):
        """ Apply categorical imputation on the given features 
        
        :param columns: List of columns names
        """
        self.cols = columns
        self.imps = [CategoricalImputer() for _ in columns]
        
    def fit(self, X, y=None):
        for col, imp in zip(self.cols, self.imps):
            imp.fit(X[col])
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for col, imp in zip(self.cols, self.imps):
            X[col] = imp.transform(X[col])
        return X
    

class MultiNumericalImputer(TransformerMixin):
    """ Pipeline transformer to impute a limited set of numerical features """
    def __init__(self, columns, **kwargs):
        """ Apply numerical imputation on the given features 
        
        :param columns: List of column names
        :param kwargs: Keyword arguments are passed to the Imputer class
        """
        self.cols = columns
        self.imp = Imputer(**kwargs)
        
    def fit(self, X, y=None):
        self.imp.fit(X[self.cols])
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X[self.cols] = self.imp.transform(X[self.cols])
        return X

    
class DummyEncoder(TransformerMixin):
    """ One-hot encode a set of categorical features of the dataset """
    def __init__(self, columns):
        """ Apply one-hot encoding to the given set of categorical features 
        
        :param columns: List of column names
        """
        self.cols = columns
        self.encs = [LabelBinarizer() for _ in columns]
    
    def fit(self, X, y=None):
        for col, enc in zip(self.cols, self.encs):
            enc.fit(X[col])
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for col, enc in zip(self.cols, self.encs):
            arr = enc.transform(X[col])
            for category, column in zip(enc.classes_, arr.T):
                newcat = '{}_{}'.format(col, re.sub('\W', '', category))
                X[newcat] = column
        return X.drop(columns=self.cols)
    
    
class OutOfFoldModel:
    """ Helper class that ensures out-of-fold sampling for stacked binary classifiers. 
    
    This class avoids that the second layer in ensemble stacking is trained on different
    samples than the training samples of the classifiers in the first layer. This is done
    by training the first layer over k folds. The holdout sets are used to generate the 
    output of the first layer.
    """
    def __init__(self, model_class, kf, **kwargs):
        """ Create an out-of-fold object from a sklearn model
        
        :param model_class: class of the sklearn model to build
        :param kf: sklearn KFold object that will be used to create the folds
        :param kwargs: Additional keyword arguments are passed to the model constructor
        """
        self.classifiers = []
        self.kf = kf
        self.model_class = model_class
        self.kwargs = kwargs
        
    def fit(self, X, y):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.ravel() if isinstance(y, pd.Series) else y
        print("Fitting model {}({})".format(
            self.model_class.__name__, 
            ', '.join('{}={}'.format(key, val) for key, val in self.kwargs.items())
        ))
        oof_train = np.zeros((X.shape[0],))

        for i, (train_index, test_index) in enumerate(self.kf):
            x_tr = X[train_index]
            y_tr = y[train_index]
            x_te = X[test_index]
            y_te = y[test_index]

            clf = self.model_class(**self.kwargs)
            clf.fit(x_tr, y_tr)
            # Calibrating the output sets the range of probabilities to [0, 1]
            calibrated = CalibratedClassifierCV(clf, cv='prefit', method='isotonic')
            calibrated.fit(x_te, y_te)
            self.classifiers.append(calibrated)

            oof_train[test_index] = calibrated.predict_proba(x_te)[:, 1]
        
        print("   y_pred:", oof_train)
        print("   AUC:", roc_auc_score(y, oof_train))
        return oof_train

    def predict(self, X):
        oof_pred_skf = None
        
        for i, clf in enumerate(self.classifiers):
            pred = clf.predict(X)
            if oof_pred_skf is None:
                oof_pred_skf = np.empty((len(self.classifiers), *pred.shape))
            oof_pred_skf[i, :] = pred
        
        return oof_pred_skf.mean(axis=0)

    def predict_proba(self, X):
        oof_pred_skf = None
        
        for i, clf in enumerate(self.classifiers):
            proba = clf.predict_proba(X)[:, 1]
            if oof_pred_skf is None:
                oof_pred_skf = np.empty((len(self.classifiers), *proba.shape))
            oof_pred_skf[i, :] = proba
        
        return oof_pred_skf.mean(axis=0)
    
    
class StackedModel(TransformerMixin):
    """ Helper class to allow ensemble stacking in a sklearn Pipeline """
    def __init__(self, first_models, second_model, n_folds=5):
        """ Create a stacked model of two layers 
        
        :param first_models: List of tuples with first element the sklearn model class
            and the second element the keyword arguments as a dictionary
        :param second_model: Tuple with first element the sklearn model class and the
            second element the keyword arguments as a dictionary
        :param n_folds: Number of folds to use for the out-of-fold models
        """
        self.first_models_def = first_models
        self.second_model_def = second_model
        self.n_folds = n_folds
        
        self._first_models = []
        self._second_model = None
        
    def fit(self, X, y):
        kf = KFold(X.shape[0], n_folds=self.n_folds)
        X_second = np.empty((y.shape[0], len(self.first_models_def)))
        for i, (model_cls, kwargs) in enumerate(self.first_models_def):
            
            oof_model = OutOfFoldModel(model_class=model_cls, kf=kf, **kwargs)
            X_second[:, i] = oof_model.fit(X, y)
            self._first_models.append(oof_model)
        
        scaler = StandardScaler()
        second_model = self.second_model_def[0](**self.second_model_def[1])
        pipeline = Pipeline(steps=[('scaler', scaler), ('model', second_model)])
        self._second_model = pipeline
        self._second_model.fit(X_second, y)
    
    def transform(self, X, y=None):
        return X
        
    def predict(self, X):
        X_second = np.empty((X.shape[0], len(self._first_models)))
        for i, model in enumerate(self._first_models):
            X_second[:, i] = model.predict_proba(X)
        return self._second_model.predict(X_second)
        
    def predict_proba(self, X):
        X_second = np.empty((X.shape[0], len(self._first_models)))
        for i, model in enumerate(self._first_models):
            X_second[:, i] = model.predict_proba(X)
        return self._second_model.predict_proba(X_second)
    

def keras_builder(dims):
    """ Builder for the keras model. 
    
    This function is parsed to the KerasClassifier constructor 
    """
    model = Sequential()
    dropout = 0.2
    assert len(dims) >= 3
    model.add(Dense(units=dims[1], activation='relu', input_dim=dims[0], kernel_constraint=max_norm(4.),
                    bias_constraint=max_norm(4.)))
    model.add(Dropout(dropout))
    for dim in dims[1:-1]:
        model.add(Dense(units=dim, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(units=dims[-1], activation='sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_pipeline():
    """
    This function should build an sklearn.pipeline.Pipeline object to train
    and evaluate a model on a pandas DataFrame. The pipeline should end with a
    custom Estimator that wraps a TensorFlow model. See the README for details.
    """
    drop_cols = ['education']
    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'native-country']

    drop_target = Drop(['target'], errors='ignore')
    drop = Drop(drop_cols)
    num_imputer = MultiNumericalImputer(columns=num_cols, strategy='mean')
    cat_imputer = MultiCategoricalImputer(columns=cat_cols)
    encoder = DummyEncoder(columns=cat_cols)
    scaler = StandardScaler()
    pca = PCA(n_components=20)
    selection = SelectKBest(k=0)
    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
    class_weight = {0: 1, 1: 4}
    classifiers = [
        (LogisticRegression, {'class_weight': class_weight, 'C': 0.9}),
        (GaussianNB, {}),
        (AdaBoostClassifier, {'n_estimators': 300, 'learning_rate': 0.85}),
        (ExtraTreesClassifier, {'n_estimators': 300, 'max_depth': 5, 'min_samples_leaf': 2, 'n_jobs': -1,
                                'class_weight': class_weight}),
        (GradientBoostingClassifier, {'n_estimators': 300, 'max_depth': 5, 'subsample': 0.9, 
                                      'min_samples_leaf': 2}),
        (RandomForestClassifier, {'n_estimators': 300, 'n_jobs': -1, 'max_depth': 15,
                                  'class_weight': class_weight}),
        (KerasClassifier, {'build_fn': keras_builder, 'dims': [88, 128, 64, 1], 'epochs': 10, 
                           'class_weight': class_weight, 'verbose': 0})
    ]
    if xgb is not None:
        classifiers.append(
            (xgb.XGBClassifier, {'n_estimators': 500, 'max_depth': 4, 'min_child_weight': 2,
                                 'gamma': 0.9, 'subsample': 0.8, 'colsample_bytree': 0.8, 
                                 'objective': 'binary:logistic', 'nthread': -1, 'scale_pos_weight': 1})
        )
    kerasmodel = StackedModel(
        first_models=classifiers,
        second_model=(RandomForestClassifier, {'n_estimators': 1000, 'n_jobs': -1, 'max_depth': 5,
                                               'min_samples_leaf': 2, 'class_weight': {0: 1, 1: 2}})
        
    )
    return Pipeline(steps=[('drop_target', drop_target), ('drop', drop), ('cat_imputer', cat_imputer),
                           ('num_imputer', num_imputer), ('encoder', encoder), ('scaler', scaler), 
                           #('feat_select', combined_features), 
                           ('model', kerasmodel)])