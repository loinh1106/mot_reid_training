from .veri import VeRi
from .aic_reid import AIC_reid, AIC_Feature_Extractor
from .dataset_loader import ImageDataset, AICDataset

__factory = {"veri": VeRi, "aic": AIC_reid}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))

    return __factory[name](*args, **kwargs)