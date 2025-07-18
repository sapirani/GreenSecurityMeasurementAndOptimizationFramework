from utils.general_consts import SummaryType
from summary_builder import NativeSummaryBuilder, SystemResourceIsolationSummaryBuilder


def summary_builder_factory(summary_type: SummaryType):
    if summary_type == SummaryType.NATIVE:
        return NativeSummaryBuilder()
    elif summary_type == SummaryType.ISOLATE_SYSTEM_RESOURCES:
        return SystemResourceIsolationSummaryBuilder()

    raise Exception("Selected summary builder is not supported")