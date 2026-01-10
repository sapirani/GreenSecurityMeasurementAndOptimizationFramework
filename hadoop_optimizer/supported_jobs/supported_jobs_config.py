from typing import List
from DTOs.hadoop.job_types import JobType


# TODO: SUPPORT DIVERSE KINDS OF JOBS (SUCH AS MONTE CARLO PI, WHICH ITS INPUT IS ALWAYS SMALL).
# TODO: SUPPORT VARYING INPUT SIZES PER JOBS (IF IT IS NECESSARY)
class SupportedJobsConfig:
    @staticmethod
    def get_all_jobs() -> List[JobType]:
        return list(JobType)

    @staticmethod
    def get_supported_input_size_gb(job_type: JobType) -> List[float]:
        return [1, 5, 10]
