from enum import Enum
from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel, Field

from scanner_trigger.trigger_sender import send_trigger


class TriggerAction(str, Enum):
    START_MEASUREMENT = "start_measurement"
    STOP_MEASUREMENT = "stop_measurement"
    STOP_PROGRAM = "stop_program"


class NodesTriggerSender(BaseModel):
    session_id_prefix: str = Field("", description="Control the session id sent to the scanner")
    number_of_datanodes: int = Field(3, gt=0, description="number of hadoop_workers")

    resource_manager_url: str = "resourcemanager-1"
    resource_manager_port: str = 65432
    namenode_url: str = "namenode-1"
    namenode_port: int = 65432
    history_server_url: str = "historyserver-1"
    history_server_port: int = 65432

    datanode_prefix: str = Field("datanode", description="Hostname prefix for each DataNode")
    datanodes_port: int = Field(65432, description="Port for each DataNode")

    @property
    def datanodes_urls(self) -> List[str]:
        return [
            f"{self.datanode_prefix}-{i}"
            for i in range(1, self.number_of_datanodes + 1)
        ]

    def get_receivers_addresses(self) -> List[Tuple[str, int]]:
        return [
            *[(datanode_url, self.datanodes_port) for datanode_url in self.datanodes_urls],
            (self.resource_manager_url, self.resource_manager_port),
            (self.namenode_url, self.namenode_port),
            (self.history_server_url, self.history_server_port)
        ]

    def _build_full_session_id(self, session_id: str):
        full_session_id = self.session_id_prefix
        if session_id:
            full_session_id += f":{session_id}" if self.session_id_prefix else session_id

        return full_session_id

    def start_measurement(
            self,
            *,
            session_id: Optional[str] = None,
            scanner_logging_extras: Optional[Dict[str, str]] = None
    ):
        send_trigger(
            TriggerAction.START_MEASUREMENT,
            self.get_receivers_addresses(),
            self._build_full_session_id(session_id),
            scanner_logging_extras
        )

    def stop_measurement(self, *, session_id: Optional[str] = None):
        send_trigger(
            TriggerAction.STOP_MEASUREMENT,
            self.get_receivers_addresses(),
            self._build_full_session_id(session_id)
        )

    def stop_program(self):
        send_trigger(
            TriggerAction.STOP_PROGRAM,
            self.get_receivers_addresses(),
            ""
        )
