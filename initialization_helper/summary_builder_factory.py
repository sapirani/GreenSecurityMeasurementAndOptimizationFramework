from general_consts import SummaryType
from summary_builder import DuduSummary, OtherSummary


def summary_builder_factory(summary_type: SummaryType):
    if summary_type == SummaryType.DUDU:
        return DuduSummary()
    elif summary_type == SummaryType.OTHER:
        return OtherSummary()

    raise Exception("Selected summary builder is not supported")