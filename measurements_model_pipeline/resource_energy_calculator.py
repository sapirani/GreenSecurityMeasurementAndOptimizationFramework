import logging
from dataclasses import asdict

from DTOs.aggregators_features.energy_model_features.process_energy_model_features import ProcessEnergyModelFeatures
from measurements_model_pipeline.sample_resources_energy import SampleResourcesEnergy

class EnergyPerResourceConsts:
    """
    This class holds constant values that represent the energy consumption per one unit of a specific resource.
    For example, the energy usage for acquiring 1 MB of RAM is 17.18 mwh.
    """
    cpu_time_seconds = 1.17
    memory_gain_mb = 0.04 # todo: change
    memory_release_mb = 0.03  # todo: change to actual number
    disk_io_read_kbytes = 0.1261034238
    disk_io_write_kbytes = 0.1324211241
    network_received_kbytes = 0.1161303828
    network_sent_kbytes = 0.005866983801

class ResourceEnergyCalculator:
    def __init__(self):
        self.__energy_per_cpu_time = EnergyPerResourceConsts.cpu_time_seconds
        self.__energy_per_gain_mb_ram = EnergyPerResourceConsts.memory_gain_mb
        self.__energy_per_release_mb_ram = EnergyPerResourceConsts.memory_release_mb
        self.__energy_per_disk_read_kb = EnergyPerResourceConsts.disk_io_read_kbytes
        self.__energy_per_disk_write_kb = EnergyPerResourceConsts.disk_io_write_kbytes
        self.__energy_per_network_received_kb = EnergyPerResourceConsts.network_received_kbytes
        self.__energy_per_network_sent_kb = EnergyPerResourceConsts.network_sent_kbytes

    def calculate_energy_consumption_per_resource(self, process_features: ProcessEnergyModelFeatures) \
            -> SampleResourcesEnergy:
        cpu_energy = self.__calculate_cpu_energy(
            process_features.cpu_usage_seconds_process)

        memory_diff = process_features.memory_mb_relative_process
        if memory_diff < 0:
            memory_energy = self.__calculate_mb_released_ram_energy(memory_diff)
        else:
            memory_energy = self.__calculate_mb_gained_ram_energy(memory_diff)

        disk_io_write_energy = self.__calculate_disk_write_kb_energy(
            process_features.disk_write_kb_process)

        disk_io_read_energy = self.__calculate_disk_read_kb_energy(
            process_features.disk_read_kb_process)

        network_received_energy = self.__calculate_network_received_kb_energy(
            process_features.network_kb_received_process)

        network_sent_energy = self.__calculate_network_sent_kb_energy(
            process_features.network_kb_sent_process)

        return SampleResourcesEnergy(
            cpu_energy_consumption=cpu_energy,
            ram_energy_consumption=memory_energy,
            disk_io_read_energy_consumption=disk_io_read_energy,
            disk_io_write_energy_consumption=disk_io_write_energy,
            network_io_received_energy_consumption=network_received_energy,
            network_io_sent_energy_consumption=network_sent_energy)

    def calculate_total_energy_by_resources(self, process_features: ProcessEnergyModelFeatures) -> float:
        energy_per_resource = self.calculate_energy_consumption_per_resource(process_features)
        per_resource_energy_sum = self.__calculate_total_energy(energy_per_resource)
        return per_resource_energy_sum

    @staticmethod
    def __calculate_total_energy(energy_per_resource: SampleResourcesEnergy) -> float:
        return sum(asdict(energy_per_resource).values())

    def calculate_relative_energy_consumption(self, process_features: ProcessEnergyModelFeatures,
                                              total_energy: float) -> SampleResourcesEnergy:
        energy_per_resource = self.calculate_energy_consumption_per_resource(process_features)
        total_energy_by_resources = self.__calculate_total_energy(energy_per_resource)
        return SampleResourcesEnergy(
            cpu_energy_consumption=ResourceEnergyCalculator.__normalize_energy_consumption(
                energy_per_resource.cpu_energy_consumption, total_energy_by_resources, total_energy),
            ram_energy_consumption=ResourceEnergyCalculator.__normalize_energy_consumption(
                energy_per_resource.ram_energy_consumption, total_energy_by_resources, total_energy),
            disk_io_read_energy_consumption=ResourceEnergyCalculator.__normalize_energy_consumption(
                energy_per_resource.disk_io_read_energy_consumption, total_energy_by_resources, total_energy),
            disk_io_write_energy_consumption=ResourceEnergyCalculator.__normalize_energy_consumption(
                energy_per_resource.disk_io_write_energy_consumption, total_energy_by_resources, total_energy),
            network_io_received_energy_consumption=ResourceEnergyCalculator.__normalize_energy_consumption(
                energy_per_resource.network_io_received_energy_consumption, total_energy_by_resources, total_energy),
            network_io_sent_energy_consumption=ResourceEnergyCalculator.__normalize_energy_consumption(
                energy_per_resource.network_io_sent_energy_consumption, total_energy_by_resources, total_energy)
        )

    def __calculate_cpu_energy(self, cpu_usage: float) -> float:
        return self.__energy_per_cpu_time * cpu_usage

    def __calculate_mb_released_ram_energy(self, mb_released: float) -> float:
        return self.__energy_per_release_mb_ram * mb_released * -1

    def __calculate_mb_gained_ram_energy(self, mb_gained: float) -> float:
        return self.__energy_per_gain_mb_ram * mb_gained

    def __calculate_disk_read_kb_energy(self, disk_read_kb: float) -> float:
        return self.__energy_per_disk_read_kb * disk_read_kb

    def __calculate_disk_write_kb_energy(self, disk_write_kb: float) -> float:
        return self.__energy_per_disk_write_kb * disk_write_kb

    def __calculate_network_received_kb_energy(self, network_kb: float) -> float:
        return self.__energy_per_network_received_kb * network_kb

    def __calculate_network_sent_kb_energy(self, network_kb: float) -> float:
        return self.__energy_per_network_sent_kb * network_kb

    @staticmethod
    def __normalize_energy_consumption(resource_energy: float, resource_sum_energy: float,
                                       model_energy_prediction: float):
        if model_energy_prediction <= 0:
            logging.warning(f"Energy prediction is negative or 0, equals to: {model_energy_prediction}")
            return 0
        elif resource_sum_energy <= 0:
            logging.warning(f"Energy sum by resources is negative or 0, equals to: {resource_sum_energy}")
            return 0

        logging.info(f"Energy prediction is positive and equals to {model_energy_prediction} mwh")
        return (model_energy_prediction / resource_sum_energy) * resource_energy
