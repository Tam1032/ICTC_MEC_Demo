import numpy as np

from typing import NamedTuple


class Block(NamedTuple):
    compute_requirement: float  # Compute requirement for the exit in CPU cycles
    #input_size: float  # Input size for the exit in MB
    accuracy: float  # Accuracy of the exit

class DNN_Model:
    def __init__(self, model_size, blocks):
        """
        Initialize the DNN model with a list of blocks.
        Each block is a tuple containing (num_layers, compute_requirement, input_size).
        """
        self.blocks = blocks  # List of block namedtuples
        self.model_size = model_size  # Model size in MB
        self.num_blocks = len(blocks)

    def get_block(self, index):
        """
        Get the block at the specified index.
        """
        if index < 0 or index >= self.num_blocks:
            raise IndexError("block index out of range")
        return self.blocks[index]

    def get_compute_requirement(self, exit_decision):
        total_computation = self.blocks[exit_decision].compute_requirement
        return total_computation
    
    def get_total_compute_requirement(self):
        total_computation = sum(block.compute_requirement for block in self.blocks)
        return total_computation

    def get_accuracy(self, exit_decision):
        accuracy = self.blocks[exit_decision].accuracy
        return accuracy


class Task:
    def __init__(self, device_id, input_size, task_type, arrival_time):
        self.device_id = device_id
        self.input_size = input_size  # in MB
        self.task_type = task_type  # DNN_Inference type
        self.arrival_time = arrival_time    

    def get_transmitted_size(self, exit_decision):
        """
        Get the transmitted size based on the offload decision.
        """
        return self.input_size

    def get_compute_requirement(self, exit_decision):
        """
        Get the compute requirement based on the offload decision.
        """
        return self.task_type.get_compute_requirement(exit_decision)
    
    def get_accuracy(self, exit_decision):
        """
        Get the expected compute requirement based on the offload decision.
        """
        return self.task_type.get_accuracy(exit_decision)
