import objax
import copy
from tqdm import tqdm

__all__ = ['convert']


def convert(model: objax.Module, save_path: str = None, do_copy = False) -> objax.Module:
    """
    Convert a train architecture RepVGG model to deploy architecture
    :param model: Objax Model
    :param save_path: path to save VarCollection to
    :param do_copy: create a copy
    :return: RepVGG model with inference architecture
    """
    if model.deploy:
        print("Model architecture is already deploy, nothing to convert")
        return model
    else:
        if do_copy:
            model = copy.deepcopy(model)
        blocks = model.get_blocks()
        blocks = tqdm(blocks)
        print("Converting model")
        for block in blocks:
            if hasattr(block, 'switch_to_deploy'):
                block.switch_to_deploy()
        if save_path is not None:
            objax.io.save_var_collection(save_path, save_path)
        return model