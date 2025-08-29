from aggregation_manager import AggregationManager
from elastic_reader import ElasticReader
from user_input.GUI.gui_date_picker import GUITimePicker
from application_logging.logging_utils import get_measurement_logger
from application_logging.handlers.elastic_handler import get_elastic_logging_handler
from consts import ES_URL, ES_USER, ES_PASS, ElasticIndex
from utils.general_consts import LoggerName, IndexName


if __name__ == '__main__':
    # TODO: SUPPORT USER FLAGS HERE
    # TODO: SUPPORT CLI TIME PICKER

    logger = get_measurement_logger(
        logger_name=LoggerName.METRICS_AGGREGATIONS,
        logger_handler=get_elastic_logging_handler(ES_USER, ES_PASS, ES_URL, IndexName.METRICS_AGGREGATIONS),
    )

    # TODO: SUPPORT VERBOSE MODE

    # handler = StreamHandler()
    # handler.setFormatter(PrettyExtrasFormatter())
    # logger.addHandler(handler)

    time_picker = GUITimePicker()
    time_picker_input = time_picker.get_input()
    print(time_picker_input)

    reader = ElasticReader(time_picker_input, [ElasticIndex.PROCESS, ElasticIndex.SYSTEM])

    # TODO: CREATE AN INTERFACE FOR ALL elastic consumers, AND LET EACH ONE CHOOSING THE RESULTS IT IS INTERESTED IN
    # TODO: (AS IT WAS DONE IN THE AGGREGATORS). EXTRACT HE AGGREGATORS AND THE RELEVANT DTOs TO SOMEWHERE ELSE

    # TODO: SPIT LOGGING FROM AGGREGATORS AND ENHANCE THE SPEED OF LOGGING (ESPECIALLY IN OFFLINE)
    aggregation_manager = AggregationManager()

    for iteration_results in reader.read():
        try:
            aggregation_manager.aggregate_iteration_raw_results(iteration_results)
        except Exception as e:
            print(
                "Warning! It seems like indexing is too slow. Consider increasing MAX_INDEXING_TIME_SECONDS")
            print("The received exception:", e)
