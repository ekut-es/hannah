from .classifier import CrossValidationStreamClassifierModule, StreamClassifierModule
from .distilling_classifier import SpeechKDClassifierModule
from .image_classifier import ImageClassifierModule
from .object_detection import ObjectDetectionModule

__all__ = [
    "CrossValidationStreamClassifierModule",
    "StreamClassifierModule",
    "SpeechKDClassifierModule",
    "ImageClassifierModule",
    "ObjectDetectionModule",
]
