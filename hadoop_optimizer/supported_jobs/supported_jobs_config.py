from pathlib import Path
from typing import List

from DTOs.hadoop.hadoop_job_definition import HadoopJobDefinition
from DTOs.hadoop.job_descriptor import JobDescriptor
from DTOs.hadoop.job_properties import JobProperties
from DTOs.hadoop.job_types import JobType


# TODO: SUPPORT DIVERSE KINDS OF JOBS (SUCH AS MONTE CARLO PI, WHICH ITS INPUT IS ALWAYS SMALL).
# TODO: SUPPORT VARYING INPUT SIZES PER JOBS (IF IT IS NECESSARY)
class SupportedJobsConfig:
    @staticmethod
    def get_all_jobs() -> List[JobType]:
        return list(JobType)

    # TODO: MAKE THIS LIST DYNAMIC (MAYBE BY A SEPARATE API ON THE TRAINING SERVER THAT OBSERVES THE DIRECTORY
    #  STRUCTURE AND UNDERSTANDS) WHAT ARE THE SUPPORTED INPUT SIZES
    @staticmethod
    def get_supported_input_size_gb(job_type: JobType) -> List[float]:
        return [1, 5, 10]

    @staticmethod
    def extract_job_properties(job_descriptor: JobDescriptor) -> JobProperties:
        # TODO:
        #   1. ENSURE THAT THE MANUAL ESTIMATE IS REASONABLE
        #   2. TRY TO THINK OF AN AUTOMATIC WAY TO EXTRACT JOB PROPERTIES
        if job_descriptor.job_type == JobType.word_count:
            return JobProperties(
                input_size_gb=job_descriptor.input_size_gb,
                cpu_bound_scale=0.5,  # moderate CPU
                io_bound_scale=0.8,  # heavy I/O
            )

        elif job_descriptor.job_type == JobType.anagrams:
            return JobProperties(
                input_size_gb=job_descriptor.input_size_gb,
                cpu_bound_scale=0.8,  # heavy CPU (sorting words)
                io_bound_scale=0.6,  # moderate I/O
            )

        elif job_descriptor.job_type == JobType.line_statistics:
            return JobProperties(
                input_size_gb=job_descriptor.input_size_gb,
                cpu_bound_scale=0.4,  # very light CPU
                io_bound_scale=0.8,  # almost pure I/O
            )

        else:
            raise ValueError(f"Unsupported job type: {job_descriptor.job_type}")

    @staticmethod
    def extract_job_definition(job_descriptor: JobDescriptor) -> HadoopJobDefinition:
        if job_descriptor.job_type == JobType.word_count:
            return HadoopJobDefinition(
                input_path=Path(f"/input/{job_descriptor.input_size_gb}_gb"),
                output_path=Path(f"/output/{job_descriptor.input_size_gb}_gb"),
                mapper_path=Path(f"/home/mapreduce/word_count/mapper.py"),
                reducer_path=Path("/home/mapreduce/word_count/reducer.py"),
            )

        elif job_descriptor.job_type == JobType.anagrams:
            return HadoopJobDefinition(
                input_path=Path(f"/input/{job_descriptor.input_size_gb}_gb"),
                output_path=Path(f"/output/{job_descriptor.input_size_gb}_gb"),
                mapper_path=Path(f"/home/mapreduce/anagrams/mapper.py"),
                reducer_path=Path("/home/mapreduce/anagrams/reducer.py"),
            )

        elif job_descriptor.job_type == JobType.line_statistics:
            return HadoopJobDefinition(
                input_path=Path(f"/input/{job_descriptor.input_size_gb}_gb"),
                output_path=Path(f"/output/{job_descriptor.input_size_gb}_gb"),
                mapper_path=Path(f"/home/mapreduce/line_statistics/mapper.py"),
                reducer_path=Path("/home/mapreduce/line_statistics/reducer.py"),
            )

        else:
            raise ValueError(f"Unsupported job type: {job_descriptor.job_type}")
