import requests
from urllib.parse import urljoin
from hadoop_optimizer.DTOs.hadoop_job_execution_config import HadoopJobExecutionConfig
from hadoop_optimizer.DTOs.job_properties import JobProperties
from hadoop_optimizer.deployment_client.consts import DEFAULT_CHOOSE_CONFIG_ENDPOINT_NAME, DEFAULT_SERVER_PORT, \
    DEFAULT_SERVER_IP


class HadoopOptimizerDeploymentClient:
    def __init__(
            self,
            server_ip: str = DEFAULT_SERVER_IP,
            server_port: int = DEFAULT_SERVER_PORT,
            choose_configuration_endpoint_name: str = DEFAULT_CHOOSE_CONFIG_ENDPOINT_NAME
    ):
        self.api_address = f"http://{server_ip}:{server_port}"
        self.choose_configuration_endpoint_name = choose_configuration_endpoint_name

    def get_optimal_configuration(self, job_properties: JobProperties) -> HadoopJobExecutionConfig:
        """
        :raises: requests.exceptions.HTTPError: 503 service unavailable
        """
        response = requests.get(
            urljoin(self.api_address, self.choose_configuration_endpoint_name),
            params=job_properties.model_dump(exclude_none=True)
        )
        response.raise_for_status()

        return HadoopJobExecutionConfig.model_validate(response.json())
