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
rules = [
        "Windows Event For Service Disabled",
        "Detect New Local Admin account",
        "ESCU Network Share Discovery Via Dir Command Rule",
        "Known Services Killed by Ransomware",
        "Non Chrome Process Accessing Chrome Default Dir",
        "Kerberoasting spn request with RC4 encryption",
        "Clop Ransomware Known Service Name"
    ]
def init_worker():
    """Initialize SplunkTools instance once per process"""
    global process_splunk

    process_splunk = SplunkTools(rules, 3, 180)

def process_chunk(chunk_data, top_logtypes):
    """Process a chunk of rows using the process-level SplunkTools instance"""
    global process_splunk
    
    chunk_X = []
    chunk_Y = {}
    
    # Initialize Y dictionary for all rules
    for rule in process_splunk.active_saved_searches:
        chunk_Y[f"rule_cpu_{rule['title']}"] = []

    # Process each row in the chunk
    for _, row in chunk_data.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        x = []
        
        # Process CPU metrics
        for rule in process_splunk.active_saved_searches:
            cpu = row[f"rule_cpu_{rule['title']}"]
            chunk_Y[f"rule_cpu_{rule['title']}"].append(cpu)
        
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
        ('wineventlog:security', '4663'), ('wineventlog:security', '4732'),
        ('wineventlog:security', '4769'), ('wineventlog:security', '5140'),
        ('wineventlog:system', '7036'), ('wineventlog:system', '7040'),
        ('wineventlog:system', '7045'), ('wineventlog:security', '4624')
    ]

    # Read and process data
    no_agent_data_path = r'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/experiments_____/no_agent_baseline/no_agent_20250131_000047.csv'
    no_agent_csv = pd.read_csv(no_agent_data_path)

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
    # pd.DataFrame(X).to_csv("X.csv")
    # pd.DataFrame(Y).to_csv("Y.csv")
    
    # Load X and Y
    X = pd.read_csv("X.csv").drop(columns=["Unnamed: 0"])
    Y = pd.read_csv("Y.csv")
    # normalize X
    X = (X - X.mean()) / X.std()
    list_top_logtypes = list(top_logtypes)
    for rule in rules:
        print("---------------")
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y[f"rule_cpu_{rule}"], test_size=0.2, random_state=42)
        regr = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=2000)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        print(f"Rule: {rule}")
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        
        # Get top 10 feature importance form mlp regressor
        # sample
        sample = X_train.sample(20)
        explainer = shap.KernelExplainer(regr.predict, sample)
        shap_values = explainer.shap_values(sample)
        plot = shap.summary_plot(shap_values, sample, feature_names=list_top_logtypes, plot_type='bar')
        
        plt.savefig(f"shap_bar_{rule}.png")
        #plot results
        # plt.scatter(y_test, y_pred)
        # plt.xlabel("True CPU")
        # plt.ylabel("Predicted CPU")
        # plt.title(f"True vs Predicted CPU for Rule: {rule}")
        # # add line
        # plt.plot([0, 5], [0, 5], color='red')
        # plt.savefig(f"true_vs_pred_{rule}.png")

        plt.close()
        print("---------------")
    
if __name__ == "__main__":
    main()