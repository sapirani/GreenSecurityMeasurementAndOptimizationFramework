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
from strategies.reward_strategy import *
from splunk_tools import SplunkTools
from unsused_scripts.datetime_manager import MockedDatetimeManager
from strategies.state_strategy import StateStrategy4
from matplotlib import pyplot as plt
import sys
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment')

from SplunkResearch.resources.section_logtypes import section_logtypes

if not ("alerts.pkl" in os.listdir() and "distributions.pkl" in os.listdir() and "time_ranges.pkl" in os.listdir() and "rules_alerts_probs.csv" in os.listdir()):
    start_time = "04/01/2024:00:00:00"
    savedsearches = ["Windows Event For Service Disabled",
                    "Detect New Local Admin account",
                    "ESCU Network Share Discovery Via Dir Command Rule",
                    "Known Services Killed by Ransomware",
                    "Non Chrome Process Accessing Chrome Default Dir",
                    "Kerberoasting spn request with RC4 encryption",
                    "Clop Ransomware Known Service Name"]
    frequency = "1"
    top_logtypes = pd.read_csv("/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/resources/top_logtypes.csv")
    top_logtypes = top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:50]
    top_logtypes = [(x[0].lower(), str(x[1])) for x in top_logtypes]
    print("num of rows in top_logtypes: ", len(top_logtypes))
    splunk = SplunkTools(savedsearches, 1, frequency)
    datetime_manager = MockedDatetimeManager(start_time)
    reward_strategy = RewardStrategy43(datetime_manager, splunk, len(savedsearches),no_agent_table_path="/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments_____/no_agent_baseline/no_agent_20241228_231350.csv" )
    state_strategy = StateStrategy4(top_logtypes)

    end_time = "06/01/2024:00:00:00"
    search_window = "1"
    section_logtypes = {k:' '.join(x[0]) for k,x in section_logtypes.items()}
        
    def gather_data(time_ranges):
        alerts = [] 
        distributions = []
        rules_alerts_probs = []
        for time_range in time_ranges:
            data = reward_strategy.get_no_agent_reward(time_range)
            alert_val = data['alert']
            relevant_row = reward_strategy.no_agent_values[(reward_strategy.no_agent_values['start_time'] == time_range[0]) & (reward_strategy.no_agent_values['end_time'] == time_range[1])]
            rules_alerts = {section_logtypes[i]:relevant_row[f'rule_alert_{i}'].values[0] for i in savedsearches}
            alerts.append(alert_val)
            distribution = splunk.get_real_distribution(*time_range)
            rules_logtypes = {section_logtypes[i]:distribution[section_logtypes[i]] for i in savedsearches if section_logtypes[i] in distribution}
            rules_alerts_prob = {k:rules_alerts[k]/rules_logtypes[k] for k in rules_alerts if k in rules_logtypes}
            rules_alerts_prob.update({k:0 for k in rules_alerts if k not in rules_logtypes})
            rules_alerts_probs.append(rules_alerts_prob)
            fake_distribution = {"no":0 for i in range(len(top_logtypes))}
            state_strategy.update_distributions(distribution, fake_distribution)
            real_state = state_strategy.get_abs_states()[0]
            distributions.append(real_state)
            if random.random() < 0.1:
                print(time_range, alert_val, real_state)
            
        return alerts, distributions, rules_alerts_probs

    time_ranges = []
    while datetime.datetime.strptime(start_time, '%m/%d/%Y:%H:%M:%S') < datetime.datetime.strptime(end_time, '%m/%d/%Y:%H:%M:%S'):
        current_end_time = datetime_manager.add_time(start_time, hours=int(search_window), minutes=0, seconds=0)
        time_ranges.append((start_time, current_end_time))
        start_time = datetime_manager.add_time(start_time, hours=int(frequency), minutes=0, seconds=0)
        # print(start_time)
        
    alerts, distributions, rules_alerts_probs = gather_data(time_ranges)
    rules_alerts_probs_df = pd.DataFrame(rules_alerts_probs)
    rules_alerts_probs_df.to_csv("rules_alerts_probs.csv")

    # dump time_ranges, alerts, distributions to a file
    with open("time_ranges.pkl", "wb") as f:
        pickle.dump(time_ranges, f)
    with open("alerts.pkl", "wb") as f:
        pickle.dump(alerts, f)
    with open("distributions.pkl", "wb") as f:
        pickle.dump(distributions, f)
else:
    with open("time_ranges.pkl", "rb") as f:
        time_ranges = pickle.load(f)
    with open("alerts.pkl", "rb") as f:
        alerts = pickle.load(f)
    with open("distributions.pkl", "rb") as f:
        distributions = pickle.load(f)
    rules_alerts_probs = pd.read_csv("rules_alerts_probs.csv").drop(columns=["Unnamed: 0"])
    # plot rules_alerts_probs in multy axes
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for i, rule in enumerate(rules_alerts_probs.columns):
        ax = axs[i//4, i%4]
        ax.plot(rules_alerts_probs[rule])
        ax.set_title(rule)
    plt.savefig("rules_alerts_probs.png")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertPredictor:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.scaler = preprocessing.StandardScaler()
    
    def prepare_data_for_lstm(self, distributions, alerts, time_ranges, test_size=0.2):
        """split alerts to sliding windows and prepare data for training"""
        X = []
        y = []
        window_size = 5
        for i in range(window_size, len(alerts)):
            X.append(alerts[i-window_size:i])
            y.append(alerts[i])
        X = np.array(X)
        X = X.astype('int')
        y = np.array(y)
        y = y.astype('int')
        # Normalize features
        # X = preprocessing.normalize(X)
        
        # Split data
        return train_test_split(X, y, test_size=test_size, random_state=121)
               
      
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
            'n_estimators': [2000],
            'max_depth': [10],
            'min_samples_split': [5],
            'min_samples_leaf': [2]
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
    
    def train_lstm_model(self, X_train, y_train):
        """Train LSTM model"""
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.callbacks import EarlyStopping
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)   
        model.fit(X_train, y_train, batch_size=64, epochs=1000, callbacks=[early_stop])
        
        self.model = model
    
    def evaluate_lstm_model(self, X_test, y_test):
        """Evaluate LSTM model performance"""
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
    # X_train, X_test, y_train, y_test = predictor.prepare_data_for_lstm(distributions, alerts, time_ranges)
    
    # Train model
    predictor.train_model(X_train, y_train)
    # predictor.train_lstm_model(X_train, y_train)
    
    # Evaluate model
    metrics_dict, y_pred = predictor.evaluate_model(X_test, y_test)
    # metrics_dict, y_pred = predictor.evaluate_lstm_model(X_test, y_test)
    
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