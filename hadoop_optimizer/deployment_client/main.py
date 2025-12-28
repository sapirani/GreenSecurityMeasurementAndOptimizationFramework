from hadoop_optimizer.DTOs.job_properties import JobProperties
from hadoop_optimizer.deployment_client.hadoop_optimizer_deployment_client import HadoopOptimizerDeploymentClient


def main():
    deployment_client = HadoopOptimizerDeploymentClient()
    job_properties = JobProperties(input_size_gb=10, cpu_bound_scale=0.5, io_bound_scale=0.5)
    print(deployment_client.get_optimal_configuration(job_properties))


if __name__ == '__main__':
    main()
