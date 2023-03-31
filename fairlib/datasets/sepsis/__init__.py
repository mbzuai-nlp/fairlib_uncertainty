from .sepsis import Sepsis

def init_data_class(dest_folder, batch_size, model_name):
    return Sepsis(dest_folder = dest_folder, batch_size=batch_size, model_name=model_name)
