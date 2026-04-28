from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN_HW5_2_1
    for k in range(num_batches + num_partitions - 1):
        schedule = []
        for j in range(num_partitions):
            i = k - j
            if 0 <= i < num_batches:
                schedule.append((i, j))
        yield schedule
    # END_HW5_2_1

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.

        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN_HW5_2_2
        batches = list(x.chunk(self.split_size, dim=0))
        num_batches = len(batches)
        num_partitions = len(self.partitions)
        for schedule in _clock_cycles(num_batches, num_partitions):
            self.compute(batches, schedule)
        return torch.cat(batches, dim=0).to(self.devices[-1])
        # END_HW5_2_2

    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker.
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN_HW5_2_2
        # Phase 1: submit all (i, j) tasks in this clock to their workers.
        for i, j in schedule:
            batch = batches[i].to(devices[j], non_blocking=True)
            partition = partitions[j]

            def closure(batch=batch, partition=partition):
                return partition(batch)

            task = Task(closure)
            self.in_queues[j].put(task)

        # Phase 2: collect results (one per submitted task on each worker).
        for i, j in schedule:
            ok, payload = self.out_queues[j].get()
            if not ok:
                exc_info = payload
                raise exc_info[1].with_traceback(exc_info[2])
            _task, result = payload
            batches[i] = result
        # END_HW5_2_2

