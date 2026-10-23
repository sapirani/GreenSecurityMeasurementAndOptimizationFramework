import subprocess
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from DTOs.hadoop.consts import Groups, HDFS_NAMENODE


# TODO: ENSURE SMOOTH INTEGRATION WITH THE GNS3 REPO WHEN COMBINING ALL INTO A MONOREPO
#  (SPECIFICALLY, OUTPUT VALIDATION)
class HadoopJobDefinition(BaseModel):

    # Support model validation with aliases
    model_config = {
        "validate_by_name": True,
        "validate_by_alias": True,
    }

    input_path: Path = Field(
        default=Path("/input"),
        alias="i",
        title=Groups.TASK_DEFINITION.value,
        description="HDFS path to the input directory"
    )

    output_path: Path = Field(
        default=Path("/output"),
        alias="o",
        title=Groups.TASK_DEFINITION.value,
        description="HDFS path to the output directory",
    )

    mapper_path: Path = Field(
        default=Path("/home") / Path("mapper.py"),
        alias="mp",
        title=Groups.TASK_DEFINITION.value,
        description="Path to the mapper implementation",
    )

    reducer_path: Path = Field(
        default=Path("/home") / Path("reducer.py"),
        alias="rp",
        title=Groups.TASK_DEFINITION.value,
        description="Path to the reducer implementation",
    )

    @field_validator("output_path", mode="after")
    def ensure_no_output_path(cls, output_path: str) -> str:
        if cls.hdfs_path_exists(Path(HDFS_NAMENODE) / Path(output_path)):
            raise FileExistsError(f"Output path already exists: {output_path}")

        return output_path

    @staticmethod
    def hdfs_path_exists(path: Path) -> bool:
        result = subprocess.run(
            ["hdfs", "dfs", "-test", "-e", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
