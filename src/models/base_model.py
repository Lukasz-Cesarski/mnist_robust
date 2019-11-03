from abc import ABC, abstractmethod


class BaseModel(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError
