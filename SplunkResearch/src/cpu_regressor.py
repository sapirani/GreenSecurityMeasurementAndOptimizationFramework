import sys
from multiprocessing import Pool, cpu_count
from functools import partial

from matplotlib import pyplot as plt
from splunk_tools import SplunkTools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import shap

# Global process-level variable to store SplunkTools instance
process_splunk = None
rules = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name",
                 'Windows AD Replication Request Initiated from Unsanctioned Location',
                 'ESCU Windows Rapid Authentication On Multiple Hosts Rule']
def init_worker():
    """Initialize SplunkTools instance once per process"""
    global process_splunk

    process_splunk = SplunkTools(rules, 1)

def process_chunk(chunk_data, top_logtypes):
    """Process a chunk of rows using the process-level SplunkTools instance"""
    global process_splunk
    
    chunk_X = []
    chunk_Y = {}
    
    # Initialize Y dictionary for all rules
    for rule in process_splunk.active_saved_searches:
        chunk_Y[f"rule_cpu_{rule}"] = []
        chunk_Y[f"rule_alert_{rule}"] = []

    # Process each row in the chunk
    for (start_time, end_time), group in chunk_data.groupby(['start_time', 'end_time']):

        x = []
        if len(group) != len(process_splunk.active_saved_searches):
            print(f"Warning: Missing data for time range {start_time} - {end_time}. Skipping this chunk.")
            continue
        # Process CPU metrics
        for rule in process_splunk.active_saved_searches:
            cpu = float(group[group['search_name'] == rule]['cpu'].values[0])
            chunk_Y[f"rule_cpu_{rule}"].append(cpu)
            alert = float(group[group['search_name'] == rule]['alert'].values[0])
            chunk_Y[f"rule_alert_{rule}"].append(alert)
        
        # Get log distribution
        log_distribution = process_splunk.get_real_distribution(start_time, end_time)
        for logtype in top_logtypes:
            if logtype in log_distribution:
                x.append(log_distribution[logtype])
            else:
                x.append(0)
        
        chunk_X.append(x)
    
    return chunk_X, chunk_Y

def main():
    # Define relevant log types
    relevant_logtypes = [
        ('wineventlog:security', '4663'), ('wineventlog:security', '4662'),('wineventlog:security', '4732'),
        ('wineventlog:security', '4769'), ('wineventlog:security', '5140'),
        ('wineventlog:system', '7036'), ('wineventlog:system', '7040'),
        ('wineventlog:system', '7045'), ('wineventlog:security', '4624')
    ]

    # Read and process data
    global_path = r'/home/shouei/GreenSecurity-FirstExperiment/experiments/baseline'
    no_agent_data_paths = [rf'{global_path}/baseline_splunk_train-v32_2880.csv']#[rf'{global_path}/no_agent_20250129_114533.csv', rf'{global_path}/no_agent_20250131_000047.csv']
    
    for no_agent_data_path in no_agent_data_paths:
        # determine a dir to save the results
        results_dir = no_agent_data_path.split("/")[-1].split(".")[0]+"_cpu_regressor_results"
        # create a dir
        import os
        os.makedirs(results_dir, exist_ok=True)
        # Load data
        no_agent_csv = pd.read_csv(no_agent_data_path)
        # no_agent_csv['start_time'] = pd.to_datetime(no_agent_csv['start_time'], format='%m/%d/%Y:%H:%M:%S')
        # no_agent_csv['end_time'] = pd.to_datetime(no_agent_csv['end_time'], format='%m/%d/%Y:%H:%M:%S')
        no_agent_csv = no_agent_csv.groupby(['search_name',"start_time", "end_time"]).aggregate({"cpu": "mean", "alert": "mean"}).reset_index()
        # sort by start time and end time
        no_agent_csv = no_agent_csv.sort_values(by=['start_time', 'end_time'])
        # Process top log types
        top_logtypes = pd.read_csv(r"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/resources/top_logtypes.csv")
        top_logtypes = top_logtypes[top_logtypes['source'].str.lower().isin(['wineventlog:security', 'wineventlog:system'])]
        top_logtypes = top_logtypes.sort_values(by='count', ascending=False)[['source', "EventCode"]].values.tolist()[:100]
        top_logtypes = [(x[0].lower(), str(x[1])) for x in top_logtypes]
        top_logtypes = set(top_logtypes) | set(relevant_logtypes)

        # # Initialize multiprocessing pool with worker initialization
        # num_cores = cpu_count()
        # pool = Pool(processes=num_cores, initializer=init_worker)

        # # Split data into chunks
        # chunk_size = max(1, len(no_agent_csv) // num_cores)
        # chunks = [no_agent_csv[i:i + chunk_size] for i in range(0, len(no_agent_csv), chunk_size)]

        # # Process chunks in parallel
        # process_chunk_partial = partial(process_chunk, top_logtypes=top_logtypes)
        # results = pool.map(process_chunk_partial, chunks)

        # # Close the pool
        # pool.close()
        # pool.join()

        # # Combine results from all chunks
        # X = []
        # Y = {}
        
        # # Initialize Y dictionary
        # first_chunk_Y = results[0][1]
        # for key in first_chunk_Y:
        #     Y[key] = []

        # # Combine all chunks
        # for chunk_X, chunk_Y in tqdm(results):
        #     X.extend(chunk_X)
        #     for key in chunk_Y:
        #         Y[key].extend(chunk_Y[key])
        # # dump X and Y
        # pd.DataFrame(X).to_csv(f"{results_dir}/X.csv")
        # pd.DataFrame(Y).to_csv(f"{results_dir}/Y.csv")
        
        # Load X and Y
        X = pd.read_csv(f"{results_dir}/X.csv").drop(columns=["Unnamed: 0"])
        Y = pd.read_csv(f"{results_dir}/Y.csv")
        # Plot y over time
        # for rule in rules:
        #     plt.plot(Y[f"rule_cpu_{rule}"])
        #     plt.title(f"CPU over time for Rule: {rule}")
        #     plt.savefig(f"{results_dir}/cpu_over_time_{rule}.png")
        #     plt.close()
        # normalize X
        X = X.apply(lambda x: x/sum(x), axis=1)
        X_alerts = Y[[f"rule_alert_{rule}" for rule in rules]]
        Y = Y[[f"rule_cpu_{rule}" for rule in rules]]
        X_alerts = X_alerts + 0.0001
        X_alerts = (X_alerts - X_alerts.mean()) / X_alerts.std()
        with_alerts = [0, 1]

        list_top_logtypes = list(top_logtypes)
        for with_alert in with_alerts:
            for rule in rules:
                print("---------------")
                if with_alert:
                    X['101'] = X_alerts[f"rule_alert_{rule}"]

                X_train, X_test, y_train, y_test = train_test_split(X, Y[f"rule_cpu_{rule}"], test_size=0.2, random_state=42)
                regr1 = RandomForestRegressor(n_estimators=100, random_state=42)
                
                regr2 = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=3000)
                for regr in [regr1, regr2]:
                    
                    regr.fit(X_train, y_train)
                    y_pred = regr.predict(X_test)
                    print(f"Rule: {rule}")
                    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
                    print(f"R2 Score: {regr.score(X_test, y_test)}")
                    # dump the model
                    import pickle
                    with open(f"{results_dir}/{regr.__class__.__name__}_{rule}_with alert = {with_alert}.pkl", 'wb') as f:
                        pickle.dump(regr, f)
                    # # Get top 10 feature importance form mlp regressor
                    # # sample
                    # sample = X_train.sample(200)
                    # explainer = shap.KernelExplainer(regr.predict, sample)
                    # shap_values = explainer.shap_values(sample)
                    # plot = shap.summary_plot(shap_values, sample, feature_names=list_top_logtypes, plot_type='bar')
                    
                    # plt.savefig(f"{regr.__class__.__name__}_shap_bar_{rule}.png")
                    # plot results
                    # plt.scatter(y_test, y_pred)
                    # plt.xlabel("True CPU")
                    # plt.ylabel("Predicted CPU")
                    # plt.title(f"True vs Predicted CPU for Rule: {rule}")
                    # # add line
                    # plt.plot([0, 10], [0, 10], color='red')
                    # # add mean squared error
                    # plt.text(0, 9, f"MSE: {mean_squared_error(y_test, y_pred)}")
                    # # add r2 score
                    # plt.text(0, 8, f"R2 Score: {regr.score(X_test, y_test)}")
                    # plt.savefig(f"{results_dir}/{regr.__class__.__name__}_true_vs_pred_{rule}_with alert = {with_alert}.png")

                    plt.close()
                    print("---------------")
    
if __name__ == "__main__":
    main()