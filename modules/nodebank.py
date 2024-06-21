import numpy as np


class NodeBank(object):
    def __init__(
        self,
        src: np.ndarray,
        dst: np.ndarray,
    ):
        r"""
        maintains a dictionary of all nodes seen so far (specified by the input src and dst)
        Parameters:
            src: source node id of the edges
            dst: destination node id of the edges
            ts: timestamp of the edges
        """
        self.nodebank = {}
        self.update_memory(src, dst)


    def update_memory(self, 
                      update_src: np.ndarray, 
                      update_dst: np.ndarray) -> None:
        r"""
        update self.memory with newly arrived src and dst
        Parameters:
            src: source node id of the edges
            dst: destination node id of the edges
        """
        for src, dst in zip(update_src, update_dst):
            if src not in self.nodebank:
                self.nodebank[src] = 1
            if dst not in self.nodebank:
                self.nodebank[dst] = 1


    def query_node(self, node: int) -> bool:
        r"""
        query if node is in the memory
        Parameters:
            node: node id to query
        Returns:
            True if node is in the memory, False otherwise
        """
        return node in self.nodebank
