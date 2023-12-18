"""
Shreyas Prasad
CS 7180 - Advanced Perception
"""

import importlib
import torch

class LDMModelLoader:
    """
    A utility class for loading LDM models from configuration and checkpoints.
    """

    def __init__(self):
        pass

    @staticmethod
    def instantiate_from_config(config):
        """
        Instantiate an LDM model from a configuration dictionary.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            object: The instantiated LDM model.
        """
        if not "target" in config:
            if config == '__is_first_stage__':
                return None
            elif config == "__is_unconditional__":
                return None
            raise KeyError("Expected key `target` to instantiate.")
        return LDMModelLoader.get_object_from_string(config["target"])(**config.get("params", dict()))

    @staticmethod
    def get_object_from_string(string, reload_module=False):
        """
        Get an object from a string representation.

        Args:
            string (str): The string representation of the object.
            reload_module (bool, optional): Whether to reload the module. Defaults to False.

        Returns:
            object: The object obtained from the string representation.
        """
        module_path, class_name = string.rsplit(".", 1)
        if reload_module:
            module_imported = importlib.import_module(module_path)
            importlib.reload(module_imported)
        return getattr(importlib.import_module(module_path, package=None), class_name)

    @staticmethod
    def load_model_from_config(config, checkpoint, verbose=False):
        """
        Load an LDM model from a configuration and checkpoint.

        Args:
            config (dict): The configuration dictionary.
            checkpoint (str): The path to the checkpoint file.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.

        Returns:
            object: The loaded LDM model.
        """
        print(f"Loading model from {checkpoint}")
        pl_sd = torch.load(checkpoint, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        state_dict = pl_sd["state_dict"]
        model = LDMModelLoader.instantiate_from_config(config.model)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0 and verbose:
            print("missing keys:")
            print(missing)
        if len(unexpected) > 0 and verbose:
            print("unexpected keys:")
            print(unexpected)

        model.cuda()
        model.eval()
        return model

