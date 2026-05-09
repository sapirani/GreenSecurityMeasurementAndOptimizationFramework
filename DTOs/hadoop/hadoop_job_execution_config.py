import argparse
import re

from pydantic import BaseModel, Field, model_validator

# TODO: UNIFY WITH THE GNS3 PROJECT
from DTOs.hadoop.consts import HUMAN_READABLE_KEY, units, CompressionCodec, GarbageCollector, Groups


def parse_size(value: str) -> int:
    """
    Parse a size string with optional units: B, KB, MB, GB (case-insensitive).
    Returns the size in bytes as an int.

    Examples:
    - "128MB" -> 134217728
    - "1G"    -> 1073741824
    - "512kb" -> 524288
    - "100"   -> 100 bytes
    """

    pattern = r"^\s*(\d+)\s*([KMGB]{1,2})?\s*$"
    match = re.match(pattern, value.strip().upper())

    if not match:
        raise argparse.ArgumentTypeError(f"Invalid size value: '{value}'")

    number, unit = match.groups()
    number = int(number)
    multiplier = units.get(unit, 1)

    return number * multiplier


class HadoopJobExecutionConfig(BaseModel):
    """
    For now, any field *must* have a default value. If you wish to add a field without a default value, add
    required=True in this field when converting this class into argparse.
    """

    # Support model validation with aliases
    model_config = {
        "validate_by_name": True,
        "validate_by_alias": True,
    }

    # Parallelism & Scheduling
    number_of_mappers: int = Field(
        default=2,
        gt=0,
        alias="m",
        title=Groups.PARALLELISM_AND_SCHEDULING.value,
        description="Number of mapper tasks",
    )

    number_of_reducers: int = Field(
        default=1,
        gt=0,
        alias="r",
        title=Groups.PARALLELISM_AND_SCHEDULING.value,
        description="Number of reducer tasks",
    )

    map_vcores: int = Field(
        default=1,
        gt=0,
        alias="mc",
        title=Groups.PARALLELISM_AND_SCHEDULING.value,
        description="Number of vCores per map task",
    )

    reduce_vcores: int = Field(
        default=1,
        gt=0,
        alias="rc",
        title=Groups.PARALLELISM_AND_SCHEDULING.value,
        description="Number of vCores per reduce task",
    )

    application_manager_vcores: int = Field(
        default=1,
        gt=0,
        alias="ac",
        title=Groups.PARALLELISM_AND_SCHEDULING.value,
        description="Number of vCores for the application master",
    )

    shuffle_copies: int = Field(
        default=5,
        gt=0,
        alias="sc",
        title=Groups.PARALLELISM_AND_SCHEDULING.value,
        description="Parallel copies per reduce during shuffle. "
                    "More copies speed up shuffle but risk saturating network or disk I/O.",
    )

    jvm_numtasks: int = Field(
        default=1,
        gt=0,
        alias="jvm",
        title=Groups.PARALLELISM_AND_SCHEDULING.value,
        description="Number of tasks per JVM to reduce JVM startup overhead. "
                    "While JVM reuse improves efficiency, "
                    "it introduces a risk of tasks affecting one another due to shared state.",
    )

    slowstart_completed_maps: float = Field(
        default=0.05,
        ge=0,
        le=1,
        alias="ssc",
        title=Groups.PARALLELISM_AND_SCHEDULING.value,
        description="Fraction of maps to finish before reduce begins. "
                    "Higher delays reduce phase but reduces load on shuffle.",
    )

    # Memory
    heap_memory_ratio: float = Field(
        default=0.8,
        gt=0,
        le=1,
        alias="hmr",
        title=Groups.MEMORY.value,
        description="Ratio of container memory allocated to the JVM heap versus non-heap memory "
                    "(e.g., stack, native buffers, etc.)",
    )

    map_memory_mb: int = Field(
        default=1024,
        gt=0,
        alias="mm",
        title=Groups.MEMORY.value,
        description="Memory per map task (MB).",
    )

    reduce_memory_mb: int = Field(
        default=1024,
        gt=0,
        alias="rm",
        title=Groups.MEMORY.value,
        description="Memory per reduce task (MB)",
    )

    application_manager_memory_mb: int = Field(
        default=1536,
        gt=0,
        alias="am",
        title=Groups.MEMORY.value,
        description="Memory for application master (MB)",
    )

    sort_buffer_mb: int = Field(
        default=100,
        gt=0,
        alias="sb",
        title=Groups.MEMORY.value,
        description="Sort buffer size (MB)",
    )

    min_split_size: int = Field(
        default=0,
        ge=0,
        alias="sm",
        title=Groups.MEMORY.value,
        description="Minimum input split size with human-readable units (B, KB, MB, GB). "
                    "Larger min split size reduces the number of map tasks, "
                    "improving startup overhead but may reduce parallelism.",
        json_schema_extra={
            HUMAN_READABLE_KEY: True,
        },
    )

    max_split_size: int = Field(
        default=128 * 1024 * 1024,
        gt=0,
        alias="sM",
        title=Groups.MEMORY.value,
        description="Maximum input split size with human-readable units (B, KB, MB, GB). "
                    "Effectively determines the number of mappers that will be used "
                    "(together with the input size).",
        json_schema_extra={
            HUMAN_READABLE_KEY: True
        },
    )

    map_min_heap_size_mb: int = Field(
        default=128,
        gt=0,
        alias="mhm",
        title=Groups.MEMORY.value,
        description="Initial heap size for each mapper’s JVM."
                    "Low values make faster starts but more garbage collector overhead.",
    )

    map_max_heap_size_mb: int = Field(
        default=384,
        gt=0,
        alias="mhM",
        title=Groups.MEMORY.value,
        description="Maximum heap size for each mapper’s JVM."
                    "Low values require more garbage collector overhead and open the risk for OutOfMemoryError "
                    "in large tasks.",
    )

    map_stack_size_kb: int = Field(
        default=1024,
        gt=0,
        alias="ms",
        title=Groups.MEMORY.value,
        description="Stack size of each mapper thread."
                    "Low values increase the risk for StackOverflowError.",
    )

    reduce_min_heap_size_mb: int = Field(
        default=128,
        gt=0,
        alias="rhm",
        title=Groups.MEMORY.value,
        description="Initial heap size for each reducer’s JVM."
                    "Low values make faster starts but more garbage collector overhead.",
    )

    reduce_max_heap_size_mb: int = Field(
        default=384,
        gt=0,
        alias="rhM",
        title=Groups.MEMORY.value,
        description="Maximum heap size for each reducer’s JVM."
                    "Low values require more garbage collector overhead and open the risk for OutOfMemoryError "
                    "in large tasks.",
    )

    reduce_stack_size_kb: int = Field(
        default=1024,
        gt=0,
        alias="rs",
        title=Groups.MEMORY.value,
        description="Stack size of each reducer thread."
                    "Low values increase the risk for StackOverflowError.",
    )

    # Shuffle & Compression
    io_sort_factor: int = Field(
        default=10,
        gt=0,
        alias="f",
        title=Groups.SHUFFLE_AND_COMPRESSION.value,
        description="Number of streams merged simultaneously during map output sort.",
    )

    should_compress: bool = Field(
        default=False,
        alias="c",
        title=Groups.SHUFFLE_AND_COMPRESSION.value,
        description="Enable compression of map outputs before shuffle. "
                    "Compression reduces network traffic at the cost of additional CPU usage.",
    )

    map_compress_codec: CompressionCodec = Field(
        default=CompressionCodec.DEFAULT,
        alias="mcc",
        title=Groups.SHUFFLE_AND_COMPRESSION.value,
        description="Compression codec for map output. "
                    "Options: " + ", ".join(f"{c.name} ('{c.value}')" for c in CompressionCodec)
    )

    # JVM & Garbage Collection Settings
    map_garbage_collector: GarbageCollector = Field(
        default=GarbageCollector.ParallelGC,
        alias="mgc",
        title=Groups.SHUFFLE_AND_COMPRESSION.value,
        description="Type of garbage collector for mappers. "
                    "Options: " + ", ".join(f"{gc.name} ('{gc.value}')" for gc in GarbageCollector)
    )

    reduce_garbage_collector: GarbageCollector = Field(
        default=GarbageCollector.ParallelGC,
        alias="rgc",
        title=Groups.SHUFFLE_AND_COMPRESSION.value,
        description="Type of garbage collector for reducers. "
                    "Options: " + ", ".join(f"{gc.name} ('{gc.value}')" for gc in GarbageCollector)
    )

    map_garbage_collector_threads_num: int = Field(
        default=1,
        gt=0,
        alias="mgct",
        title=Groups.JVM_AND_GC.value,
        description="Number of threads the mapper JVM uses for parallel garbage collection."
                    "Should be compatible with the mapper's vcores allocation",
    )

    reduce_garbage_collector_threads_num: int = Field(
        default=1,
        gt=0,
        alias="rgct",
        title=Groups.JVM_AND_GC.value,
        description="Number of threads the reducer JVM uses for parallel garbage collection."
                    "Should be compatible with the reducer's vcores allocation",
    )

    @model_validator(mode="after")
    def check_split_sizes_validity(self) -> "HadoopJobExecutionConfig":
        if not self.min_split_size <= self.max_split_size:
            raise ValueError(
                f"`max_split_size` ({self.max_split_size}) must be greater "
                f"or equal to `min_split_size` ({self.min_split_size})"
            )
        return self

    def __str__(self) -> str:
        return self.__repr_str__('\n')
