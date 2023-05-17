"""
EdgeBank is a simple strong baseline for dynamic link prediction
it predicts the existence of edges based on their history of occurrence

Reference:
    - https://github.com/fpour/DGB/tree/main
"""


import numpy as np
from collections import defaultdict


class EdgeBankPredictor_MemoryUpdate(object):
    def __init__(
        self,
        memory_mode: str = 'unlimited',  # could be `unlimited` or `fixed_time_window`
        time_window_ratio: float = 0.15
    ):
        assert memory_mode in ['unlimited', 'fixed_time_window'], "Invalide memory mode for EdgeBank!"
        self.memory_mode = memory_mode
        if self.memory_mode == 'fixed_time_window':
            self.time_window_ratio = time_window_ratio
        else:
            self.time_window_ratio = -1
        
        self.memory = None
        self.pos_prob = 1.0
        self.neg_prob = 0.0

    def _update_memory(self, history_src: np.ndarray, history_dst: np.ndarray, history_ts: np.ndarray):
        r"""
        generate the current and correct state of the memory with the observed edges so far
        note that historical edges may include training, validation, and already observed test edges
        """
        if self.memory_mode == 'unlimited':
            self.memory = self._get_unlimited_memory(history_src, history_dst)
        elif self.memory_mode == 'fixed_time_window':
            self.memory = self._get_time_window_memory(history_src, history_dst, history_ts)
        else:
            raise ValueError("Invalide memory mode!")
    
    def _get_unlimited_memory(self, history_src: np.ndarray, history_dst: np.ndarray):
        r"""
        return an updated version of memory when the memory_mode is `unlimited`
        the memory contains every edges seen so far
        """
        # edge_memories = set((src, dst) for src, dst in zip(history_src, history_dst))  # not efficient!
        edge_memories = {}
        for src, dst in zip(history_src, history_dst):
            if (src, dst) not in edge_memories:
                edge_memories[(src, dst)] = 1
        
        return edge_memories

    def _get_time_window_memory(self, history_src: np.ndarray, history_dst: np.ndarray, history_ts: np.ndarray):
        r"""
        return an updated version of memory when the memory_mode is `fixed_time_window`
        the memory contains the edges seen in a fixed length recent history
        """
        time_window_start_time = np.quantile(history_ts, 1 - self.time_window_ratio)
        time_window_end_time = max(history_ts)

        memory_mask = np.logical_and(history_ts <= time_window_end_time, history_ts >= time_window_start_time)
        edge_memories = self._get_unlimited_memory(history_src[memory_mask], history_dst[memory_mask])
        return edge_memories

    def predict_link_proba(self, history_src: np.ndarray, history_dst: np.ndarray, history_ts: np.ndarray, 
                           query_src: np.ndarray, query_dst: np.ndarray):
        r"""
        get the probability of the link existence for the query batch, given the history of the observed edges
        """
        # first, update the memory
        self._update_memory(history_src, history_dst, history_ts)

        # then, make prediction
        link_pred_proba = []
        for src, dst in zip(query_src, query_dst):
            if (src, dst) in self.memory:
                link_pred_proba.append(self.pos_prob)
            else:
                link_pred_proba.append(self.neg_prob)
        
        return np.array(link_pred_proba)



class EdgeBankPredictor(object):
    def __init__(
        self,
        history_src: np.ndarray,
        history_dst: np.ndarray,
        history_ts: np.ndarray,
        memory_mode: str = 'unlimited',  # could be `unlimited` or `fixed_time_window`
        time_window_ratio: float = 0.15,
    ):
        assert memory_mode in ['unlimited', 'fixed_time_window'], "Invalide memory mode for EdgeBank!"
        self.memory_mode = memory_mode
        if self.memory_mode == 'fixed_time_window':
            self.time_window_ratio = time_window_ratio
        else:
            self.time_window_ratio = -1
        
        self.pos_prob = 1.0
        self.neg_prob = 0.0
        self._update_memory(history_src, history_dst, history_ts)

    def _update_memory(self, history_src: np.ndarray, history_dst: np.ndarray, history_ts: np.ndarray):
        r"""
        generate the current and correct state of the memory with the observed edges so far
        note that historical edges may include training, validation, and already observed test edges
        """
        print("DEBUG: self.memory_mode:", self.memory_mode)
        if self.memory_mode == 'unlimited':
            self.memory = self._get_unlimited_memory(history_src, history_dst)
        elif self.memory_mode == 'fixed_time_window':
            self.memory = self._get_time_window_memory(history_src, history_dst, history_ts)
        else:
            raise ValueError("Invalide memory mode!")
    
    def _get_unlimited_memory(self, history_src: np.ndarray, history_dst: np.ndarray):
        r"""
        return an updated version of memory when the memory_mode is `unlimited`
        the memory contains every edges seen so far
        """
        # edge_memories = set((src, dst) for src, dst in zip(history_src, history_dst))  # not efficient!
        edge_memories = {}
        for src, dst in zip(history_src, history_dst):
            if (src, dst) not in edge_memories:
                edge_memories[(src, dst)] = 1
        
        return edge_memories

    def _get_time_window_memory(self, history_src: np.ndarray, history_dst: np.ndarray, history_ts: np.ndarray):
        r"""
        return an updated version of memory when the memory_mode is `fixed_time_window`
        the memory contains the edges seen in a fixed length recent history
        """
        time_window_start_time = np.quantile(history_ts, 1 - self.time_window_ratio)
        time_window_end_time = max(history_ts)

        memory_mask = np.logical_and(history_ts <= time_window_end_time, history_ts >= time_window_start_time)
        edge_memories = self._get_unlimited_memory(history_src[memory_mask], history_dst[memory_mask])
        return edge_memories

    def predict_link_proba(self, query_src: np.ndarray, query_dst: np.ndarray):
        r"""
        get the probability of the link existence for the query batch, given the history of the observed edges
        """
        # then, make prediction
        link_pred_proba = []
        for src, dst in zip(query_src, query_dst):
            if (src, dst) in self.memory:
                link_pred_proba.append(self.pos_prob)
            else:
                link_pred_proba.append(self.neg_prob)
        
        return np.array(link_pred_proba)
    

