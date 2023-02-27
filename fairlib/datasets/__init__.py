from . import moji
from . import bios
from . import coloredMNIST
from . import COMPAS
from . import TP_POS
from . import Adult
from . import MSCOCO
from . import imSitu
from . import RoB

name2class = {
    "moji":moji,
    "bios":bios,
    "rob":RoB,
    "coloredmnist":coloredMNIST,
    "compas":COMPAS,
    "tp_pos":TP_POS,
    "adult":Adult,
    "coco":MSCOCO,
    "imsitu":imSitu,
}

def prepare_dataset(name, dest_folder,batch_size=32):
    if name in name2class.keys():
        data_class = name2class[name.lower()]
        initialized_class = data_class.init_data_class(dest_folder=dest_folder, batch_size=batch_size)
        initialized_class.prepare_data()
    else:
        print("Unknown dataset, please double check.")