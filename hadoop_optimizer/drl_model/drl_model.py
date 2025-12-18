from typing import Optional
from DTOs.aggregated_results_dtos.iteration_aggregated_results import IterationAggregatedResults
from DTOs.raw_results_dtos.iteration_info import IterationRawResults
from elastic_reader.elastic_consumers.abstract_elastic_consumer import AbstractElasticConsumer
from hadoop_optimizer.DTOs.job_properties import JobProperties
from hadoop_optimizer.drl_model.drl_state import DRLState


class DRLModel(AbstractElasticConsumer):
    def __init__(self, drl_state: DRLState):
        self.drl_state = drl_state

    def consume(
            self,
            iteration_raw_results: IterationRawResults,
            iteration_aggregation_results: Optional[IterationAggregatedResults]
    ):
        print("Inside DRL model")
        self.drl_state.update_state(iteration_raw_results, iteration_aggregation_results)

    def determine_best_job_configuration(self, job_properties: JobProperties):
        print(job_properties)
        drl_state = self.drl_state.retrieve_state_entries(job_properties)

        print("state shape:", drl_state.shape, ", is index unique:", drl_state.index.is_unique)
        print(drl_state.to_string())
        # -D
        # mapreduce.job.maps = {self.number_of_mappers}
        # -D
        # mapreduce.job.reduces = {self.number_of_reducers}
        # -D
        # mapreduce.job.heap.memory - mb.ratio = {self.heap_memory_ratio}
        # -D
        # mapreduce.map.memory.mb = {self.map_memory_mb}
        # -D
        # mapreduce.reduce.memory.mb = {self.reduce_memory_mb}
        # -D
        # yarn.app.mapreduce.am.resource.mb = {self.application_manager_memory_mb}
        # -D
        # mapreduce.map.cpu.vcores = {self.map_vcores}
        # -D
        # mapreduce.reduce.cpu.vcores = {self.reduce_vcores}
        # -D
        # yarn.app.mapreduce.am.resource.cpu - vcores = {self.application_manager_vcores}
        # -D
        # mapreduce.task.io.sort.mb = {self.sort_buffer_mb}
        # -D
        # mapreduce.task.io.sort.factor = {self.io_sort_factor}
        # -D
        # mapreduce.map.output.compress = {str(self.should_compress).lower()}
        # -D
        # mapreduce.map.output.compress.codec = {self.map_compress_codec.value}
        # -D
        # mapreduce.input.fileinputformat.split.minsize = {self.min_split_size}
        # -D
        # mapreduce.input.fileinputformat.split.maxsize = {self.max_split_size}
        # -D
        # mapreduce.reduce.shuffle.parallelcopies = {self.shuffle_copies}
        # -D
        # mapreduce.job.jvm.numtasks = {self.jvm_numtasks}
        # -D
        # mapreduce.job.reduce.slowstart.completedmaps = {self.slowstart_completed_maps}
        # -D
        # mapreduce.map.java.opts = "-Xms{self.map_min_heap_size_mb}m -Xmx{self.map_max_heap_size_mb}m -Xss{self.map_stack_size_kb}k -XX:+{self.map_garbage_collector.value} -XX:ParallelGCThreads={self.map_garbage_collector_threads_num} -Xloggc:/var/log/hadoop/gc_logs/map_gc.log"
        # -D
        # mapreduce.reduce.java.opts = "-Xms{self.reduce_min_heap_size_mb}m -Xmx{self.reduce_max_heap_size_mb}m -Xss{self.reduce_stack_size_kb}k -XX:+{self.reduce_garbage_collector.value} -XX:ParallelGCThreads={self.reduce_garbage_collector_threads_num} -Xloggc:/var/log/hadoop/gc_logs/map_gc.log"

        # return best_configuration, reward_of_state