import requests
from urllib.parse import urljoin
from DTOs.hadoop.hadoop_job_execution_config import HadoopJobExecutionConfig
from DTOs.hadoop.job_descriptor import JobDescriptor
from hadoop_optimizer.training_client.consts import DEFAULT_CHOOSE_CONFIG_ENDPOINT_NAME, DEFAULT_SERVER_PORT, \
    DEFAULT_SERVER_IP
from DTOs.hadoop.training_run_job_response import TrainingJobRunResponse


class HadoopOptimizerTrainingClient:
    def __init__(
            self,
            server_ip: str = DEFAULT_SERVER_IP,
            server_port: int = DEFAULT_SERVER_PORT,
            run_job_endpoint_name: str = DEFAULT_CHOOSE_CONFIG_ENDPOINT_NAME
    ):
        self.api_address = f"http://{server_ip}:{server_port}"
        self.run_job_endpoint_name = run_job_endpoint_name

    def run_job(
        self,
        job_descriptor: JobDescriptor,
        execution_configuration: HadoopJobExecutionConfig,
    ) -> TrainingJobRunResponse:
        """
        :raises:
            1.  requests.exceptions.HTTPError: 422 not implemented (typically when selected job execution config is invalid)
            2.  requests.exceptions.HTTPError: 500 internal server error (typically when Hadoop job can't run for some reason)
            3.  requests.exceptions.HTTPError: 501 not implemented (typically when Hadoop is not installed)
            4.  requests.exceptions.HTTPError: 503 gateway timout (when job execution has passed time limit)
        """
        response = requests.post(
            urljoin(self.api_address, self.run_job_endpoint_name),
            params=job_descriptor.model_dump(),
            json=execution_configuration.model_dump(),
        )
        response.raise_for_status()

        return TrainingJobRunResponse.model_validate(response.json())
