from DTOs.aggregators_features.energy_model_features.process_energy_model_features import ProcessEnergyModelFeatures
from measurements_model.sample_resources_energy import SampleResourcesEnergy


class ResourceEnergyCalculator:
    def __init__(self, energy_per_cpu_time: float, energy_per_gain_mb_ram: float,
                 energy_per_release_mb_ram: float, energy_per_disk_read_kb: float,
                 energy_per_disk_write_kb: float, energy_per_network_received_kb: float,
                 energy_per_network_sent_kb: float):
        self.__energy_per_cpu_time = energy_per_cpu_time
        self.__energy_per_gain_mb_ram = energy_per_gain_mb_ram
        self.__energy_per_release_mb_ram = energy_per_release_mb_ram
        self.__energy_per_disk_read_kb = energy_per_disk_read_kb
        self.__energy_per_disk_write_kb = energy_per_disk_write_kb
        self.__energy_per_network_received_kb = energy_per_network_received_kb
        self.__energy_per_network_sent_kb = energy_per_network_sent_kb

    def calculate_relative_energy_consumption(self, process_features: ProcessEnergyModelFeatures,
                                              total_energy: float) -> SampleResourcesEnergy:
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

        per_resource_energy_sum = cpu_energy + memory_energy + disk_io_write_energy + disk_io_read_energy + network_received_energy + network_sent_energy
        return SampleResourcesEnergy(
            cpu_energy_consumption=ResourceEnergyCalculator.__normalize_energy_consumption(
                cpu_energy, per_resource_energy_sum, total_energy),
            ram_energy_consumption=ResourceEnergyCalculator.__normalize_energy_consumption(
                memory_energy, per_resource_energy_sum, total_energy),
            disk_io_read_energy_consumption=ResourceEnergyCalculator.__normalize_energy_consumption(
                disk_io_read_energy, per_resource_energy_sum, total_energy),
            disk_io_write_energy_consumption=ResourceEnergyCalculator.__normalize_energy_consumption(
                disk_io_write_energy, per_resource_energy_sum, total_energy),
            network_io_received_energy_consumption=ResourceEnergyCalculator.__normalize_energy_consumption(
                network_received_energy, per_resource_energy_sum, total_energy),
            network_io_sent_energy_consumption=ResourceEnergyCalculator.__normalize_energy_consumption(
                network_sent_energy, per_resource_energy_sum, total_energy)
        )

    def __calculate_cpu_energy(self, cpu_usage: float) -> float:
        return self.__energy_per_cpu_time * cpu_usage

    def __calculate_mb_released_ram_energy(self, mb_usage: float) -> float:
        return self.__energy_per_release_mb_ram * mb_usage * -1

    def __calculate_mb_gained_ram_energy(self, mb_usage: float) -> float:
        return self.__energy_per_gain_mb_ram * mb_usage

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
        if model_energy_prediction <= 0 or resource_sum_energy <= 0:
            return 0
        return (model_energy_prediction / resource_sum_energy) * resource_energy
