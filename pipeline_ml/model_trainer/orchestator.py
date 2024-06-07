# model_trainer/__init__.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
import numpy as np

class ModelTrainer:
    def __init__(self, data):
        """
        Initialiser l'optimiseur de modèle avec la sélection de l'appareil.
        """
        self.data = data
        X = self.data.drop(['Value X', 'Timestamp'], axis=1)  # Assuming 'Timestamp' is not useful for training
        y = self.data['Value X']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(),
            'xgboost': XGBRegressor(),  # Enabling GPU support for XGBoost
            'lightgbm': LGBMRegressor()  # Enabling GPU support for LightGBM
        }

    def train_models(self):
        results = {}
        for name, model in self.models.items():
            param_grid = self.get_param_grid(name)
            mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
            grid_search = GridSearchCV(model, param_grid, cv=7, scoring=mape_scorer, n_jobs=1)
            grid_search.fit(self.X_train, self.y_train)
            results[name] = grid_search.best_estimator_
        return results

    def get_param_grid(self, model_name):
        if model_name == 'linear_regression':
            return {}
        elif model_name == 'random_forest':
            return {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        elif model_name == 'xgboost':
            return {
                'n_estimators': [100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 2, 3]  # Adding this to help prevent overfitting
            }
        elif model_name == 'lightgbm':
            return {
                'n_estimators': [100, 150],
                'learning_rate': [0.01, 0.1],
                'max_depth': [7, 10],
                'num_leaves': [31, 50],  # Nombre de feuilles dans un arbre
                # 'min_data_in_leaf': [20, 40, 60],  # Adding this to help prevent overfitting
                'min_gain_to_split': [0.001, 0.01]  # Adding this to help prevent overfitting
            }

    def execute_training(self):
        return self.train_models()

