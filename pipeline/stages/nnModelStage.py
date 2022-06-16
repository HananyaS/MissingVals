from mlModelStage import MLModelStage
from models.abstractNN import AbstractNN
from torch.utils.data import DataLoader
import torch


class NNModelStage(MLModelStage):
    _additional_tasks = ["extract_one_before_last_layer", "forward"]

    def __init__(
        self,
        **kwargs,
    ):
        self.ml_tasks = self._tasks.copy()
        self._tasks.extend(self._additional_tasks)
        super().__init__(**kwargs)

    def _run(self, *args, **kwargs):
        if self.task in self.ml_tasks:
            return super()._run(*args, **kwargs)

        if self.task == "extract_one_before_last_layer":
            return self._extract_one_before_last_layer(*args, **kwargs)

        if self.task == "forward":
            return self._forward(*args, **kwargs)

        raise ValueError(f"Unknown task:\t{self.task}, Available tasks: {self._tasks}")

    @staticmethod
    def _extract_one_before_last_layer(model: AbstractNN, loader: DataLoader):
        model.eval()
        first_batch = True

        with torch.no_grad():
            for data in loader:
                input_data, _ = model.transform_input(data)
                output_ = model.forward_one_before_last_layer(*input_data)
                if first_batch:
                    output = output_
                    first_batch = False

                else:
                    output = torch.cat((output, output_), 0)

        return output

    @staticmethod
    def _forward(model: AbstractNN, loader: DataLoader):
        model.eval()
        first_batch = True

        with torch.no_grad():
            for data in loader:
                input_data, _ = model.transform_input(data)
                output_ = model(*input_data)
                if first_batch:
                    output = output_
                    first_batch = False

                else:
                    output = torch.cat((output, output_), 0)

        return output

    def __str__(self):
        return f"Neural Network Mod elStage {self.id}\t{self.name}"
