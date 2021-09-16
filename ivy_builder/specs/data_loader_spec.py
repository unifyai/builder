# global
import abc

# local
from ivy_builder.specs.spec import Spec
from ivy_builder.specs.spec import locals_to_kwargs
from ivy_builder.specs.dataset_spec import DatasetSpec


class DataLoaderSpec(Spec, abc.ABC):

    def __init__(self,
                 dataset_spec: DatasetSpec,
                 **kwargs) -> None:
        """
        base class for storing general parameters which define the way in which the data loader loads the dataset
        """
        kw = locals_to_kwargs(locals())
        super().__init__(dataset_spec=dataset_spec,
                         **kwargs)
        self._kwargs = kw
