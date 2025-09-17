from DTOs.aggregators_features.energy_model_features.hardware_energy_model_features import HardwareEnergyModelFeatures


class HardwareExtractor:
    @staticmethod
    def extract() -> HardwareEnergyModelFeatures:
        return HardwareEnergyModelFeatures(
            PC_type="Mobile Device",
            PC_manufacturer="Dell Inc.",
            system_family="Latitude",
            machine_type="AMD64",
            device_name="MININT-NT4GD33",
            operating_system="Windows",
            operating_system_release="10",
            operating_system_version="10.0.19045",
            processor_name="Intel64 Family 6 Model 140 Stepping 1, GenuineIntel",
            processor_physical_cores=4,
            processor_total_cores=8,
            processor_max_frequency=1805,
            processor_min_frequency=0,
            total_ram=15.732791900634766,
            physical_disk_name="NVMe Micron 2450 NVMe 512GB",
            physical_disk_manufacturer="NVMe",
            physical_disk_model="Micron 2450 NVMe 512GB",
            physical_disk_media_type="SSD",
            logical_disk_name="NVMe Micron 2450 NVMe 512GB",
            logical_disk_manufacturer="NVMe",
            logical_disk_model="Micron 2450 NVMe 512GB",
            logical_disk_disk_type="Fixed",
            logical_disk_partition_style="GPT",
            logical_disk_number_of_partitions=5,
            physical_sector_size=512,
            logical_sector_size=512,
            bus_type="RAID",
            file_system="NTFS",
            design_battery_capacity=61970,
            fully_charged_battery_capacity=47850)
