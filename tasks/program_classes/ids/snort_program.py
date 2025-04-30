from tasks.program_classes.ids.ids_program import IDSProgram


class SnortProgram(IDSProgram):
    def get_program_name(self):
        return "Snort IDS"

    def get_command(self) -> str:
        base_command = f"snort -q -l {self.log_dir} -A fast -c {self.configuration_file_path} "
        if self.interface_name is not None:
            return base_command + f"-i {self.interface_name}"

        elif self.pcap_list_dirs:
            return base_command + f'--pcap-list="{" ".join(self.pcap_list_dirs)}"'

        # return f"snort -q -l {self.log_dir} -i {self.interface_name} -A fast -c {self.configuration_file_path}"
