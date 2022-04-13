import kornia.augmentation as A

from .registry import registry

# Intensity Transformations
registry.register(A.RandomMotionBlur)
registry.register(A.ColorJiggle)
registry.register(A.RandomBoxBlur)
registry.register(A.RandomChannelShuffle)
registry.register(A.RandomEqualize)
registry.register(A.RandomGrayscale)
registry.register(A.RandomGaussianBlur)
registry.register(A.RandomGaussianNoise)
registry.register(A.RandomMotionBlur)
registry.register(A.RandomPosterize)
registry.register(A.RandomSharpness)
registry.register(A.RandomSolarize)

# Color Transformations
registry.register(A.CenterCrop)
registry.register(A.RandomAffine)
registry.register(A.RandomCrop)
registry.register(A.RandomErasing)
registry.register(A.RandomElasticTransform)
registry.register(A.RandomFisheye)
registry.register(A.RandomHorizontalFlip)
registry.register(A.RandomInvert)
registry.register(A.RandomPerspective)
registry.register(A.RandomResizedCrop)
registry.register(A.RandomRotation)
registry.register(A.RandomVerticalFlip)
registry.register(A.RandomThinPlateSpline)
