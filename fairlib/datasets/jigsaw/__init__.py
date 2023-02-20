from .jigsaw import Jigsaw

def init_data_class(dest_folder, batch_size):
    return Jigsaw(dest_folder = dest_folder, batch_size=batch_size)
