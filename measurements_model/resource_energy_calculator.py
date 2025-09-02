class ResourceEnergyCalculator:
    def __init__(self, energy_per_cpu: float, energy_per_gb_ram: float,
                 energy_per_disk_read_kb: float, energy_per_disk_write_kb: float,
                 energy_per_network_received_kb: float, energy_per_network_write_kb: float):
        self.__energy_per_cpu = energy_per_cpu
        self.__energy_per_gb_ram = energy_per_gb_ram
        self.__energy_per_disk_read_kb = energy_per_disk_read_kb
        self.__energy_per_disk_write_kb = energy_per_disk_write_kb
        self.__energy_per_network_received_kb = energy_per_network_received_kb
        self.__energy_per_network_write_kb = energy_per_network_write_kb

    def __calculate_energy_for_resource(self, energy_per_resource_consumption: float, resource_consumption: float) -> float:
        return energy_per_resource_consumption * resource_consumption

    def calculate_cpu_energy(self, cpu_usage: float) -> float:
        return self.__calculate_energy_for_resource(self.__energy_per_cpu, cpu_usage)

    def calculate_gb_ram_energy(self, gb_usage: float) -> float:
        return self.__calculate_energy_for_resource(self.__energy_per_gb_ram, gb_usage)

    def calculate_disk_read_kb_energy(self, disk_read_kb: float) -> float:
        return self.__calculate_energy_for_resource(self.__energy_per_disk_read_kb, disk_read_kb)

    def calculate_disk_write_kb_energy(self, disk_write_kb: float) -> float:
        return self.__calculate_energy_for_resource(self.__energy_per_disk_write_kb, disk_write_kb)

    def calculate_network_received_kb_energy(self, network_kb: float) -> float:
        return self.__calculate_energy_for_resource(self.__energy_per_network_received_kb, network_kb)

    def calculate_network_sent_kb_energy(self, network_kb: float) -> float:
        return self.__calculate_energy_for_resource(self.__energy_per_network_write_kb, network_kb)

    def per_resource_energy_sum(self, resource_energy: float, resource_sum_energy: float, model_energy_prediction: float):
        return (model_energy_prediction / resource_sum_energy) * resource_energy