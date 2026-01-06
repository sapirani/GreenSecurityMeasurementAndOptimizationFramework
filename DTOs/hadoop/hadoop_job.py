import re
import shlex
from pydantic import BaseModel
import argparse
import inspect
from enum import Enum
from typing import List, Type, Dict, Any
from DTOs.hadoop.consts import HDFS_NAMENODE, HUMAN_READABLE_KEY, GENERAL_GROUP
from DTOs.hadoop.hadoop_job_definition import HadoopJobDefinition
from DTOs.hadoop.hadoop_job_execution_config import parse_size, HadoopJobExecutionConfig


# TODO: ENSURE SMOOTH INTEGRATION WITH THE GNS3 REPO WHEN COMBINING ALL INTO A MONOREPO
#  (SPECIFICALLY, FROM ARGPARSE, TO ARGPARSE, AND AUTOMATIC EXPERIMENTS)
class HadoopJob(BaseModel):
    job_definition: HadoopJobDefinition
    job_execution_config: HadoopJobExecutionConfig

    @classmethod
    def from_argparse(cls, args: argparse.Namespace) -> "HadoopJob":
        data = vars(args)
        kwargs = {}

        for field_name, field in cls.model_fields.items():
            annotation = field.annotation

            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                sub_model_field_names = annotation.model_fields.keys()
                sub_model_data = {
                    key: val for key, val in data.items() if key in sub_model_field_names
                }
                kwargs[field_name] = annotation.model_validate(sub_model_data)

        return cls(**kwargs)

    @staticmethod
    def _to_argparse_add_enum_argument(
            group,
            field_default: Enum,
            flags: List[str],
            help_text: str,
            arg_type: Type[Enum]
    ):
        choices = [e.name.upper() for e in arg_type]
        group.add_argument(
            *flags,
            type=str.upper,  # parse upper-case user input
            choices=choices,
            default=field_default.name.upper(),
            help=f"{help_text} (options: {', '.join(choices)}, default: {field_default.name.upper()})"
        )

    @staticmethod
    def _to_argparse_add_human_readable_argument(
            group,
            field_default: Enum,
            flags: List[str],
            help_text: str
    ):
        group.add_argument(
            *flags,
            type=parse_size,
            default=field_default,
            help=f"{help_text} (accepts 256MB, 1G, etc., default: {field_default} bytes)"
        )

    @staticmethod
    def _to_argparse_add_boolean_argument(
            group,
            field_default: Enum,
            flags: List[str],
            help_text: str
    ):
        if field_default is True:
            group.add_argument(*flags, action="store_false", help=f"{help_text} (default: True)")
        else:
            group.add_argument(*flags, action="store_true", help=f"{help_text} (default: False)")

    @staticmethod
    def _to_argparse_add_general_argument(
            group,
            field_default: Enum,
            flags: List[str],
            help_text: str,
            arg_type: Type
    ):
        group.add_argument(
            *flags,
            type=arg_type,
            default=field_default,
            help=f"{help_text} (default: {field_default})"
        )

    @staticmethod
    def _is_enum_argument(arg_type) -> bool:
        return inspect.isclass(arg_type) and issubclass(arg_type, Enum)

    @staticmethod
    def _is_human_readable_argument(metadata) -> bool:
        return metadata.get(HUMAN_READABLE_KEY, False)

    @classmethod
    def iter_fields(cls):
        for field in cls.model_fields.values():
            annotation = field.annotation
            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                yield from annotation.model_fields.items()

    @classmethod
    def to_argparse(cls) -> argparse.ArgumentParser:
        """
        Converts the pydantic configuration into argparse to be used as CLI.
        This function uses the defaults defined in each field as the default values in the argparse
        (instance values are not taken in consideration).
        """
        parser = argparse.ArgumentParser(description="A Python wrapper for Hadoop job configuration")

        groups = {}
        for name, field in cls.iter_fields():
            meta = field.json_schema_extra or {}
            group_name = field.title if field.title else GENERAL_GROUP
            if group_name not in groups:
                groups[group_name] = parser.add_argument_group(group_name)
            group = groups[group_name]

            short_flag = f"-{field.alias}" if field.alias else None
            flags = [f"--{name}"]
            if short_flag:
                flags.insert(0, short_flag)

            help_text = field.description if field.description else ""
            field_default = field.default
            arg_type = field.annotation

            if cls._is_enum_argument(arg_type):  # Handle Enums (case-insensitive)
                cls._to_argparse_add_enum_argument(group, field_default, flags, help_text, arg_type)
            elif cls._is_human_readable_argument(meta):  # Handle human-readable sizes
                cls._to_argparse_add_human_readable_argument(group, field_default, flags, help_text)
            elif arg_type is bool:  # Handle booleans
                cls._to_argparse_add_boolean_argument(group, field_default, flags, help_text)
            else:  # Handle other types (int, float, str, etc.)
                cls._to_argparse_add_general_argument(group, field_default, flags, help_text, arg_type)

        return parser

    @classmethod
    def format_user_selection(cls, user_selection: Dict[str, Any]) -> str:
        """
        This function returns a formatted table, with 2 columns - field name and value.
        For example:
                              Field            | Value
        ---------------------------------------|----------
          application_manager_memory_mb  (am)  | 256
          input_path                     (i)   | /input
          map_memory_mb                  (mm)  | 256
          number_of_mappers              (m)   | 2
          number_of_reducers             (r)   | 1
          reduce_memory_mb               (rm)  | 256

          The data is taken from the dictionary argument that should represent the fields the user explicitly selected.
          The keys are the field names and the values are the user's selection.
        """

        def add_alias(key: str) -> str:
            alias = cls.model_fields[key].alias
            return f"({alias})" if alias and alias != key else ""

        def get_longest_key_size():
            return max(len(key) + len(add_alias(key)) for key in user_selection)

        longest_key_size = get_longest_key_size()
        first_column_width = longest_key_size + 6  # padding

        # Header row
        header = (
            "Modified by the user:\n\n"
            f"{'Field'.rjust(first_column_width // 2).ljust(first_column_width)}| Value\n"
            f"{'-' * first_column_width}|{'-' * 10}"
        )

        # Body rows
        rows = [
            f"  {(key.ljust(first_column_width - 8) + add_alias(key)).ljust(first_column_width - 2)}| {value}"
            for key, value in sorted(user_selection.items())
        ]

        return header + "\n" + "\n".join(rows)

    def get_hadoop_job_args(self) -> List[str]:
        """
        :return: A ready-to-use list of strings, which can be fed directly to subprocess.Popen, subprocess.run, etc.
        Creating a subprocess using this return value will run the distributed Hadoop job using the parameters
        configured in this class.
        """
        job_str = str(self)
        cleaned_cmd = re.sub(r"\s+", " ", job_str.strip())
        return shlex.split(cleaned_cmd)

    def __str__(self) -> str:
        return f"""
hadoop jar /opt/hadoop-3.4.1/share/hadoop/tools/lib/hadoop-streaming-3.4.1.jar
  -D mapreduce.job.maps={self.job_execution_config.number_of_mappers}
  -D mapreduce.job.reduces={self.job_execution_config.number_of_reducers}
  -D mapreduce.job.heap.memory-mb.ratio={self.job_execution_config.heap_memory_ratio}
  -D mapreduce.map.memory.mb={self.job_execution_config.map_memory_mb}
  -D mapreduce.reduce.memory.mb={self.job_execution_config.reduce_memory_mb}
  -D yarn.app.mapreduce.am.resource.mb={self.job_execution_config.application_manager_memory_mb}
  -D mapreduce.map.cpu.vcores={self.job_execution_config.map_vcores}
  -D mapreduce.reduce.cpu.vcores={self.job_execution_config.reduce_vcores}
  -D yarn.app.mapreduce.am.resource.cpu-vcores={self.job_execution_config.application_manager_vcores}
  -D mapreduce.task.io.sort.mb={self.job_execution_config.sort_buffer_mb}
  -D mapreduce.task.io.sort.factor={self.job_execution_config.io_sort_factor}
  -D mapreduce.map.output.compress={str(self.job_execution_config.should_compress).lower()}
  -D mapreduce.map.output.compress.codec={self.job_execution_config.map_compress_codec.value}
  -D mapreduce.input.fileinputformat.split.minsize={self.job_execution_config.min_split_size}
  -D mapreduce.input.fileinputformat.split.maxsize={self.job_execution_config.max_split_size}
  -D mapreduce.reduce.shuffle.parallelcopies={self.job_execution_config.shuffle_copies}
  -D mapreduce.job.jvm.numtasks={self.job_execution_config.jvm_numtasks}
  -D mapreduce.job.reduce.slowstart.completedmaps={self.job_execution_config.slowstart_completed_maps}
  -D mapreduce.map.java.opts="-Xms{self.job_execution_config.map_min_heap_size_mb}m -Xmx{self.job_execution_config.map_max_heap_size_mb}m -Xss{self.job_execution_config.map_stack_size_kb}k -XX:+{self.job_execution_config.map_garbage_collector.value} -XX:ParallelGCThreads={self.job_execution_config.map_garbage_collector_threads_num} -Xloggc:/var/log/hadoop/gc_logs/map_gc.log"
  -D mapreduce.reduce.java.opts="-Xms{self.job_execution_config.reduce_min_heap_size_mb}m -Xmx{self.job_execution_config.reduce_max_heap_size_mb}m -Xss{self.job_execution_config.reduce_stack_size_kb}k -XX:+{self.job_execution_config.reduce_garbage_collector.value} -XX:ParallelGCThreads={self.job_execution_config.reduce_garbage_collector_threads_num} -Xloggc:/var/log/hadoop/gc_logs/map_gc.log"

  -input {HDFS_NAMENODE}{self.job_definition.input_path}
  -output {HDFS_NAMENODE}{self.job_definition.output_path}
  -mapper {self.job_definition.mapper_path}
  -reducer {self.job_definition.reducer_path}
  -file {self.job_definition.mapper_path}
  -file {self.job_definition.reducer_path}
"""
