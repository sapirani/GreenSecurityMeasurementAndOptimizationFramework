from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertPredictor:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.scaler = preprocessing.StandardScaler()
        
    def prepare_data(self, distributions, alerts, test_size=0.2):
        """Prepare and split data for training"""
        X = np.array(distributions)
        y = np.array(alerts)
        
        # Normalize features
        X = preprocessing.normalize(X)
        
        # Split data
        return train_test_split(X, y, test_size=test_size, random_state=121)
    
    def train_model(self, X_train, y_train):
        """Train model with hyperparameter tuning"""
        param_grid = {
            'n_estimators': [100, 200, 1000],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4]
        }
        
        base_model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=4
        )
        
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        metrics_dict = {
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        
        logger.info("Model Performance Metrics:")
        for metric, value in metrics_dict.items():
            logger.info(f"{metric}: {value:.4f}")
            
        return metrics_dict, y_pred
    
    def plot_predictions(self, y_test, y_pred, save_path=None):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Alert Rates')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_residuals(self, y_test, y_pred, save_path=None):
        """Plot residuals"""
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_feature_importance(self, feature_names, save_path=None):
        """Plot top feature importances"""
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        top_features = importances.nlargest(10, 'importance')
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_features)), top_features['importance'])
        plt.xticks(range(len(top_features)), top_features['feature'], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
        return top_features
    
    def save_model(self, filename):
        """Save model and scaler"""
        model_path = self.model_dir / f"{filename}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        logger.info(f"Model saved to {model_path}")
        
    def load_model(self, filename):
        """Load model and scaler"""
        model_path = self.model_dir / f"{filename}.pkl"
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
        logger.info(f"Model loaded from {model_path}")

def main():
    # Load your data
    with open("alerts.pkl", "rb") as f:
        alerts = pickle.load(f)
    with open("distributions.pkl", "rb") as f:
        distributions = pickle.load(f)
    with open("time_ranges.pkl", "rb") as f:
        time_ranges = pickle.load(f)
        
    # Create predictor instance
    predictor = AlertPredictor()
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(distributions, alerts)
    
    # Train model
    predictor.train_model(X_train, y_train)
    
    # Evaluate model
    metrics_dict, y_pred = predictor.evaluate_model(X_test, y_test)
    
    # Generate plots
    predictor.plot_predictions(y_test, y_pred, 'predictions_plot.png')
    predictor.plot_residuals(y_test, y_pred, 'residuals_plot.png')
    
    # Load logtypes and plot feature importance
    top_logtypes = pd.read_csv("/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/resources/top_logtypes.csv")
    top_logtypes = top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:300]
    feature_names = [f"{x[0].lower()}_{x[1]}" for x in top_logtypes]
    top_features = predictor.plot_feature_importance(feature_names, 'feature_importance.png')
    
    # Save model
    predictor.save_model(f"alert_predictor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Print top features
    logger.info("\nTop 10 Important Features:")
    print(top_features)

if __name__ == "__main__":
    main()