from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import pytz
from datetime import datetime
import plotly.graph_objects as go
import joblib  # Import joblib for model saving 

class ModelEvaluator:
    def __init__(self, models, X_train, X_test, y_train, y_test):
        self.models = models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = []
        # Configuration des couleurs et du style
        self.background_color = 'rgba(50, 50, 50, 1)'  # Dark gray
        self.text_color = 'white'
        self.curve_colors = [
            'rgba(65, 105, 225, 0.8)',  # Blue
            'rgba(244, 164, 96, 0.8)',  # Salmon
            'rgba(205, 92, 92, 0.8)',   # Red
            'rgba(20, 92, 92, 0.8)',    # Green
            'rgba(205, 9, 92, 0.8)'     # Magenta
        ]
    def evaluate_models(self):
        # Préparation du premier graphique Plotly pour comparer les prédictions avec les vraies valeurs
        fig_predictions = go.Figure()
        fig_predictions.add_trace(go.Scatter(x=list(range(len(self.y_test))), y=self.y_test,
                                            mode='lines+markers', name='True Values', marker=dict(color='RoyalBlue')))

        # Préparation du deuxième graphique Plotly pour l'entraînement et le test
        fig_performance = go.Figure()
        path_parent = f"pipeline_ml/output_models/output_models_{datetime.now(pytz.timezone('Africa/Casablanca')).strftime('%Y-%m-%d %H:%M:%S').replace(':','_').replace(' ','_')}/"
        for name, model in self.models.items():
            model_dir = os.path.join(path_parent, name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Predictions 
            test_predictions = model.predict(self.X_test)
            train_predictions = model.predict(self.X_train)
            
            # Metrics
            train_mse = mean_squared_error(self.y_train, train_predictions)
            test_mse = mean_squared_error(self.y_test, test_predictions)
            train_mae = mean_absolute_error(self.y_train, train_predictions)
            test_mae = mean_absolute_error(self.y_test, test_predictions)
            train_mape = mean_absolute_percentage_error(self.y_train, train_predictions)
            test_mape = mean_absolute_percentage_error(self.y_test, test_predictions)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            
            new_row = pd.DataFrame({
                'Model': [name],
                'Train MSE': [train_mse],
                'Test MSE': [test_mse],
                'Train MAE': [train_mae],
                'Test MAE': [test_mae],
                'Train RMSE': [train_rmse],
                'Test RMSE': [test_rmse],
                'Train MAPE': [train_mape],
                'Test MAPE': [test_mape],
                'Train/Test Size': [f"{len(self.X_train)}/{len(self.X_test)}"],
                'Params': [model.get_params()]
            })
            self.results.append(new_row)

            # Ajout des prédictions au premier graphique
            fig_predictions.add_trace(go.Scatter(x=list(range(len(test_predictions))), y=test_predictions,
                                                mode='lines+markers', name=f'{name}_Predictions', marker=dict(size=5)))
            # Ajout des prédictions au premier graphique
            fig_predictions.add_trace(go.Scatter(x=list(range(len(test_predictions))), y=self.y_test - test_predictions,fill='tozeroy',
                                                mode='lines+markers', name=f'{name}_diff_True-Pred', marker=dict(size=5)))
            list_graphs_train = [train_mape,train_mae,train_rmse]
            list_graphs_test = [test_mape,test_mae,test_rmse]
            for metric_value,metric_name in zip(list_graphs_train,['mape','mae','rmse']) :
                fig_performance.add_trace(go.Bar(
                    x=[metric_name],  # Nom de la métrique comme axe x
                    y=[metric_value],  # Valeur de la métrique comme axe y
                    name=f'{name}_{metric_name}_train'
                ))
            for metric_value,metric_name in zip(list_graphs_test,['mape','mae','rmse']) :
                fig_performance.add_trace(go.Bar(
                    x=[metric_name],  # Nom de la métrique comme axe x
                    y=[metric_value],  # Valeur de la métrique comme axe y
                    name=f'{name}_{metric_name}_test'
                ))
                
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(10, 5))
                # Créer un graphique à barres avec les noms des caractéristiques sur l'axe des x
                feature_importances = model.feature_importances_
                indices = np.argsort(feature_importances)[::-1]  # Trier les caractéristiques par importance
                plt.bar(range(len(feature_importances)), feature_importances[indices], color='b')
                plt.xticks(range(len(feature_importances)), np.array(self.X_train.columns)[indices], rotation='vertical')
                plt.title(f'Feature Importance for {name}')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.tight_layout()  # Ajuster le layout pour s'assurer que les étiquettes sont visibles
                plt.savefig(os.path.join(model_dir, f'{name}_feature_importance.png'))
                plt.close()  # Fermer la figure pour libérer la mémoire

            # SHAP values (for tree-based models)
            if 'tree' in name:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(self.X_test)
                shap.summary_plot(shap_values, self.X_test, show=False)
                plt.savefig(os.path.join(model_dir, f'{name}_shap_values.png'))
                plt.close()  
            # save results of model:
            new_row.to_csv(os.path.join(model_dir, f"{name}_"
                f"results_{datetime.now(pytz.timezone('Africa/Casablanca')).strftime('%Y-%m-%d %H:%M:%S').replace(':','_').replace(' ','_')}.csv"), index=False)
        
        # Configuration et sauvegarde des graphiques
        fig_predictions.update_layout(
            title='Comparaison des Prédictions de Tous les Modèles avec les Vraies Valeurs',
            plot_bgcolor=self.background_color,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color)
            )
        fig_predictions.write_html(os.path.join(path_parent, "all_models_predictions.html"))

        fig_performance.update_layout(
            title='Performance de l\'Entraînement et du Test des Modèles',
            xaxis_title='Étape',
            yaxis_title='MAPE-MAE-RMSE',
            legend_title='Modèles',
            plot_bgcolor=self.background_color,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color)
            )
        fig_performance.write_html(os.path.join(path_parent, "models_training_testing_performance.html"))        
        
        # Save model using joblib
        joblib.dump(model, os.path.join(model_dir, f'{name}_model.joblib'))
        # Save all models results to a CSV file
        results_df = pd.concat(self.results)
        results_df.to_csv(os.path.join(path_parent,
                f"All_models_results_{datetime.now(pytz.timezone('Africa/Casablanca')).strftime('%Y-%m-%d %H:%M:%S').replace(':','_').replace(' ','_')}.csv"), index=False)
        self.X_train.to_csv(os.path.join(path_parent, f"train_set_X.csv"), index=False)
        self.y_train.to_csv(os.path.join(path_parent, f"train_set_y.csv"), index=False)
        self.X_test.to_csv(os.path.join(path_parent, f"test_set_X.csv"), index=False)
        self.y_test.to_csv(os.path.join(path_parent, f"test_set_y.csv"), index=False)