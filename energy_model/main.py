import pandas as pd

from energy_model.energy_prediction_model import EnergyPredictionModel

if __name__ == "__main__":
    energy_model = EnergyPredictionModel()

    full_df = pd.read_csv(r"C:\Users\Administrator\Desktop\GreenSecurityAll\framework_code\energy_model\dataset_creation\full_dataset_system_based.csv", index_col=0)
    model = energy_model.build_energy_model(full_df)
    print("Done")