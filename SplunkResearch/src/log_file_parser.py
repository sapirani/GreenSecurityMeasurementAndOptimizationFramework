import re
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        log_lines = file.readlines()

    episode_data = []
    current_episode = None
    current_fake_distribution = None
    last_duration_timestamp = None  
    last_duration_value = None  
    duration_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - duration value: (\d+\.\d+)")  
    dist2_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - dist2: \[(.*?)\]")  
    # Process each line  
    for line in log_lines:  
        duration_match = duration_pattern.search(line)  
        if duration_match:  
            # If a duration line is found, update the last seen duration info  
            last_duration_timestamp = duration_match.group(1)  
            last_duration_value = float(duration_match.group(2))
        
        dist2_match = dist2_pattern.search(line)  
        if dist2_match and last_duration_timestamp:  
            # If a dist2 line is found, and it comes after a duration line, print the result  
            dist2_timestamp = dist2_match.group(1)  
            dist2_values = dist2_match.group(2)
            dist2_values = [float(x) for x in dist2_values.strip('[]').split()]  
            print(f"After Timestamp: {last_duration_timestamp}, Duration: {last_duration_value}")  
            print(f"Timestamp: {dist2_timestamp}, Dist2: {dist2_values}")  
            current_episode = {'timestamp': last_duration_timestamp, 'duration': last_duration_value, 'energy': None, 'distribution': dist2_values}
            # Reset the last seen duration info to avoid matching subsequent dist2 lines  
            last_duration_timestamp = None  
            last_duration_value = None
            episode_data.append(current_episode)
            
    # for line in log_lines:
    #     if "dist2" in line:
    #         distribution_match = re.search(r'dist2: (\[.*\])', line)
    #         if distribution_match:
    #             current_episode = {'timestamp': None, 'duration': None, 'energy': None, 'distribution': None}

    #             distribution = distribution_match.group(1).strip('[]').split()
    #             distribution = [float(x) for x in distribution]
    #             current_episode['distribution'] = distribution

                
    #     elif "INFO - energy value" in line:
    #         energy_match = re.search(r'energy value: (\d+\.\d+)', line)
    #         if energy_match and current_episode:
    #             current_episode['energy'] = float(energy_match.group(1))

    #     elif "INFO - duration value" in line:
    #         duration_match = re.search(r'duration value: (\d+\.\d+)', line)
    #         if duration_match and current_episode:
    #             current_episode['duration'] = float(duration_match.group(1))
    #         if current_episode:
    #             episode_data.append(current_episode)


    print(len(episode_data))
    return episode_data

def prepare_data(episodes):
    X = [episode['distribution'] for episode in episodes]
    y_duration = np.array([episode['duration'] for episode in episodes])
    y_energy = np.array([episode['energy'] for episode in episodes])

    return np.array(X), y_duration, y_energy

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f'Cross-validated R-squared: {np.mean(cv_scores)}')
    return model, scaler


def train_nn_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=500, batch_size=128, validation_split=0.2, verbose=0)

    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    return model, scaler

def main():
    file_path = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments/exp_20240318_023133/log_train.txt'
    episodes = parse_log_file(file_path)
    X, y_duration, y_energy = prepare_data(episodes)
    train_test_split(X, y_duration, test_size=0.2, random_state=42)
    model, scaler = train_model(X, y_duration)
    nn_model, nn_scaler = train_nn_model(X, y_duration)

    
if __name__ == "__main__":
    main()