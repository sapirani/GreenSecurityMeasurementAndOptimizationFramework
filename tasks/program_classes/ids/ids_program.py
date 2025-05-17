from tasks.program_classes.abstract_program import ProgramInterface


class IDSProgram(ProgramInterface):
    def __init__(self, interface_name, pcap_list_dirs, log_dir, configuration_file_path=None,
                 installation_dir=r"C:\Program Files"):
        super().__init__()
        self.interface_name = interface_name
        self.pcap_list_dirs = pcap_list_dirs
        self.log_dir = log_dir
        self.installation_dir = installation_dir
        self.configuration_file_path = configuration_file_path
