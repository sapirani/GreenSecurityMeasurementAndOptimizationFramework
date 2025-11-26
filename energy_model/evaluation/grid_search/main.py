import pandas as pd
from dependency_injector import containers, providers
from dependency_injector.providers import Provider

from energy_model.configs.columns import ProcessColumns
from energy_model.energy_model_parameters import PROCESS_SYSTEM_DF_PATH
from energy_model.evaluation.grid_search.models_config import REGRESSION_MODELS_WITH_PARAMETERS
from energy_model.pipelines.grid_search_pipeline_executor import GridSearchPipelineExecutor
from energy_model.pipelines.pipeline_utils import extract_x_y

class Container(containers.DeclarativeContainer):
    grid_search_pipeline: Provider[GridSearchPipelineExecutor] = providers.Factory(GridSearchPipelineExecutor, REGRESSION_MODELS_WITH_PARAMETERS)

if __name__ == '__main__':
    container = Container()

    dataset = pd.read_csv(PROCESS_SYSTEM_DF_PATH, index_col=0)

    X, y = extract_x_y(dataset, target_column=ProcessColumns.ENERGY_USAGE_PROCESS_COL)

    best_model = container.grid_search_pipeline().run_grid_search(X, y)
