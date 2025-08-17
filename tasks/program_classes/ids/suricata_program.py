from utils.general_consts import IDSType
from tasks.program_classes.ids.ids_program import IDSProgram


class SuricataProgram(IDSProgram):
    def get_program_name(self):
        return "Suricata IDS"

    def get_command(self):
        return rf"& '{self.installation_dir}\{IDSType.SURICATA}\{IDSType.SURICATA.lower()}.exe' -i {self.interface_name} -l '{self.installation_dir}\{IDSType.SURICATA}\{self.log_dir}'"
