from abc import ABC, abstractmethod


class Stage(ABC):
    def __init__(self, name: str, task: str, n_id: int = 0, run_kwargs: dict = {}):
        self.name = name
        self.id = n_id
        self.task = task
        self.run_kwargs = run_kwargs

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def set_id(self, n_id: int):
        self.id = n_id
        return self

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def __str__(self):
        return f"Stage {self.id}\t{self.name}"

    def __copy__(self):
        return self.__class__(
            name=self.name, n_id=self.id, task=self.task, run_kwargs=self.run_kwargs
        )

    @classmethod
    def empty(cls):
        class EmptyStage(cls):
            def run(self, x):
                return x

        return EmptyStage("Empty", 0)
