from . import moji
from . import bios
from . import coloredMNIST
from . import COMPAS
from . import TP_POS
from . import Adult
from . import MSCOCO
from . import imSitu
from . import RoB
from . import sepsis
from . import jigsaw


name2class = {
    "moji":moji,
    "bios":bios,
    "rob":RoB,
    "sepsis":sepsis,
    "coloredmnist":coloredMNIST,
    "compas":COMPAS,
    "tp_pos":TP_POS,
    "adult":Adult,
    "coco":MSCOCO,
    "imsitu":imSitu,
    "jigsaw":jigsaw,
}

def prepare_dataset(name, dest_folder, model_name='bert-base-cased', batch_size=32):
    if name in name2class.keys():
        data_class = name2class[name.lower()]
        initialized_class = data_class.init_data_class(dest_folder=dest_folder, batch_size=batch_size, model_name=model_name)
        initialized_class.prepare_data()
    else:
        print("Unknown dataset, please double check.")