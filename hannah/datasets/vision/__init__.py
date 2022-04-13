from .cifar import Cifar10Dataset
from .fake import FakeDataset
from .kvasir import KvasirCapsuleDataset
from .kvasir_anomaly import KvasirCapsuleAnomalyDataset
from .kvasir_unlabeled import KvasirCapsuleUnlabeled

__all__ = [
    "KvasirCapsuleDataset",
    "FakeDataset",
    "Cifar10Dataset",
    "KvasirCapsuleUnlabeled",
    "KvasirCapsuleAnomalyDataset",
]
