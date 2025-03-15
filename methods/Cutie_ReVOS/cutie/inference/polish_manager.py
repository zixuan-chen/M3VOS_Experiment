import logging
from omegaconf import DictConfig
from typing import List, Dict
import torch

from cutie.inference.object_manager import ObjectManager
from cutie.inference.kv_memory_store import KeyValueMemoryStore
from cutie.model.cutie import CUTIE
from cutie.model.utils.memory_utils import *

log = logging.getLogger()

class PolishManager:
    """
    Reversly polish segementations every certain steps
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.use_reverse_polish = cfg.use_reverse_polish
        if self.use_reverse_polish:
            self.polish_frames = self.reverse_polish.polish_frames
        
        self.frames = []


    