from enum import Enum
from typing import Optional

GENERAL_GROUP = "General"
HUMAN_READABLE_KEY = "human_readable"
HDFS_NAMENODE = "hdfs://namenode-1:9000"
units = {
    "B": 1,
    "KB": 1024,
    "K": 1024,
    "MB": 1024 ** 2,
    "M": 1024 ** 2,
    "GB": 1024 ** 3,
    "G": 1024 ** 3
}


class CompressionCodec(str, Enum):
    DEFAULT = "org.apache.hadoop.io.compress.DefaultCodec"
    SNAPPY = "org.apache.hadoop.io.compress.SnappyCodec"
    GZIP = "org.apache.hadoop.io.compress.GzipCodec"
    LZO = "com.hadoop.compression.lzo.LzoCodec"
    BZIP2 = "org.apache.hadoop.io.compress.BZip2Codec"
    LZ4 = "org.apache.hadoop.io.compress.Lz4Codec"

    @classmethod
    def _missing_(cls, value: str) -> Optional["CompressionCodec"]:
        """
        This function is called when you try to instantiate an enum with a string value that does not appear in the
        enum values possibilities.
        The function search for compatible field names and return that field if it found one.
        """
        if isinstance(value, str):
            for member in cls:
                if member.name.lower() == value.lower(): # noqa: we are inheriting from str and Enum
                    return member   # noqa: we are inheriting from str and Enum

        return super()._missing_(value)


class GarbageCollector(str, Enum):
    SerialGC = "UseSerialGC"
    ParallelGC = "UseParallelGC"
    ConcMarkSweepGC = "UseConcMarkSweepGC"
    G1GC = "UseG1GC"

    @classmethod
    def _missing_(cls, value: str) -> Optional["GarbageCollector"]:
        """
        This function is called when you try to instantiate an enum with a string value that does not appear in the
        enum values possibilities.
        The function search for compatible field names and return that field if it found one.
        """
        if isinstance(value, str):
            for member in cls:
                if member.name.lower() == value.lower(): # noqa: we are inheriting from str and Enum
                    return member   # noqa: we are inheriting from str and Enum

        return super()._missing_(value)


class Groups(str, Enum):
    TASK_DEFINITION = "Task Definition Settings"
    PARALLELISM_AND_SCHEDULING = "Parallelism & Scheduling Settings"
    MEMORY = "Memory Settings"
    SHUFFLE_AND_COMPRESSION = "Shuffle & Compression Settings"
    JVM_AND_GC = "JVM & Garbage Collection Settings"
