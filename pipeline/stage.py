from abc import ABC, abstractmethod


class Stage(ABC):
    _tasks = []

    def __init_subclass__(cls, _tasks=(), **kwargs):
        super().__init_subclass__(**kwargs)
        cls._tasks = cls._tasks + _tasks

    def __init__(
        self,
        name: str,
        task: str,
        n_id: int = 0,
        run_kwargs: dict = {},
        input_from: dict = None,
        store_in: str = None,
        force_store: bool = True,
        *args,
        **kwargs,
    ):
        assert (
            task in self._tasks
        ), f"Unknown task:\t{task}, available tasks: {self._tasks}"

        self.name = name
        self.id = n_id
        self.task = task
        self.run_kwargs = run_kwargs
        self.input_from = input_from
        self.store_in = store_in
        self.force_store = force_store

    def run(self, *args, **kwargs):
        return self._run(*args, **kwargs)

    @abstractmethod
    def _run(self, *args, **kwargs):
        raise NotImplementedError

    def set_id(self, n_id: int):
        self.id = n_id
        return self

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def __str__(self):
        return f"Stage {self.id}\t{self.name}"

    @classmethod
    def empty(cls):
        class EmptyStage(cls):
            def run(self, x):
                return x

        return EmptyStage("Empty", 0)
