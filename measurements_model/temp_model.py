import pandas as pd

from measurements_model.energy_model import EnergyModel

if __name__ == "__main__":
    model = EnergyModel()
    model.initialize_model()
    model.predict(pd.DataFrame([[1, 2, 3], [4, 5, 6]]))