from abc import ABC, abstractmethod

import base

class BaseModel(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError

    @classmethod
    def save_dir(cls):
        return base.get_model_save_dir(cls.name())
