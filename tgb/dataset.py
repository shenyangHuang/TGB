import os
import os.path as osp
import random
import shutil
import sys
import zipfile
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from clint.textui import progress


class Data:
    """Enumerate the possible tautomers of the current molecule.
    Args:
        mol: The molecule whose state we should enumerate.
        n_variants: The maximum amount of molecules that should be returned.
        max_transforms: Set the maximum number of transformations to be applied. This limit is usually
            hit earlier than the n_variants limit and leads to a more linear scaling
            of CPU time with increasing number of tautomeric centers (see Sitzmann et al.).
        reassign_stereo: Whether to reassign stereo centers.
        remove_bond_stereo: Whether stereochemistry information is removed from double bonds involved in tautomerism.
            This means that enols will lose their E/Z stereochemistry after going through tautomer enumeration because
            of the keto-enolic tautomerism.
        remove_sp3_stereo: Whether stereochemistry information is removed from sp3 atoms involved in tautomerism. This
            means that S-aminoacids will lose their stereochemistry after going through tautomer enumeration because
            of the amido-imidol tautomerism.
    """

    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)
