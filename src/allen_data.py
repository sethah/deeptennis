import torch

from allennlp.data.fields import Field


class ImageField(Field[torch.Tensor]):
    def __init__(self,
                 label: Union[torch.Tensor],
                 label_namespace: str = '',
                 skip_indexing: bool = False) -> None:
        pass

