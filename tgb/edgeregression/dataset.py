from typing import Optional
from typing import cast
from typing import Union
from typing import List
from typing import overload
from typing import Literal

class EdgeRegressionDataset(object):
    def __init__(
        self, 
        name: str, 
        root: Optional[str] = 'dataset', 
        ):
        r"""Base Dataset class for edge regression tasks.
        Args:
            name: name of the dataset
            root: root directory to store the dataset folder
        """
        self.name = name ## original name
        self.root = root

    def pre_process(self):
        r"""Pre-process the dataset.
        """
        raise NotImplementedError
    
