# https://github.com/phaserblast/ComfyUI-DGXSparkSafetensorsLoader
# Copyright (c) 2025 Phaserblast. All rights reserved.
# Released under the Apache 2.0 license.
import gc
import logging
import torch
import folder_paths
import comfy
import comfy.model_management

# https://github.com/foundation-model-stack/fastsafetensors
from fastsafetensors import fastsafe_open,SafeTensorsFileLoader,SingleGroup

# Global registry tracking models loaded by DGXSparkSafetensorsLoader.
# Keyed by model_name, stores fastsafetensors handles and the ModelPatcher.
_dgx_registry = {}
_load_counter = 0


def _cleanup_model(model_name):
    """Free all memory for a model loaded by DGXSparkSafetensorsLoader."""
    if model_name not in _dgx_registry:
        return False

    entry = _dgx_registry.pop(model_name)
    model_patcher = entry["model_patcher"]
    fb = entry["fb"]
    loader_ref = entry["loader"]

    # Remove from ComfyUI's current_loaded_models so it doesn't track a dead model
    to_remove = []
    for i, loaded in enumerate(comfy.model_management.current_loaded_models):
        lm_model = loaded.model
        if lm_model is None:
            continue
        # Match direct reference or clones whose parent is our model
        if lm_model is model_patcher or getattr(lm_model, 'parent', None) is model_patcher:
            to_remove.append(i)
    for i in reversed(to_remove):
        try:
            lm = comfy.model_management.current_loaded_models[i]
            if lm.model_finalizer is not None:
                lm.model_finalizer.detach()
                lm.model_finalizer = None
            lm.real_model = None
        except Exception:
            pass
        comfy.model_management.current_loaded_models.pop(i)

    # Clear model parameters first to release references to fastsafetensors buffers.
    # The tensors are views into the fastsafetensors file buffer, so we must break
    # all references before closing fb.
    if hasattr(model_patcher, 'model') and model_patcher.model is not None:
        if hasattr(model_patcher.model, 'diffusion_model'):
            for name, param in list(model_patcher.model.diffusion_model.named_parameters()):
                try:
                    param.data = torch.empty(0, device='cpu')
                except Exception:
                    pass
            for name, buf in list(model_patcher.model.diffusion_model.named_buffers()):
                try:
                    buf.data = torch.empty(0, device='cpu')
                except Exception:
                    pass

    # Close fastsafetensors handles (frees underlying GPU memory)
    try:
        fb.close()
    except Exception:
        pass
    try:
        loader_ref.close()
    except Exception:
        pass

    del entry, model_patcher, fb, loader_ref
    gc.collect()
    torch.cuda.empty_cache()

    logging.info(f"[DGXSpark] Unloaded model: {model_name}")
    return True


class DGXSparkSafetensorsLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The filename of the .safetensors model to load."}),
                "device": (["cuda:0"],{"default": "cuda:0", "tooltip": "The device to which the model will be copied."}),
            }
        }
        
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a .safetensors file directly into memory using NVIDIA GPUDirect on DGX Spark."

    @classmethod
    def IS_CHANGED(cls, model_name, device):
        # If model was unloaded (not in registry), return NaN to force re-execution.
        # NaN != any cached value, so ComfyUI will re-run the loader.
        if model_name not in _dgx_registry:
            return float("nan")
        return _dgx_registry[model_name]["load_id"]

    def load_model(self, model_name, device):
        global _load_counter

        # If already loaded, return the existing model (avoids costly reload)
        if model_name in _dgx_registry:
            return (_dgx_registry[model_name]["model_patcher"],)

        dev = torch.device(device)
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        
        # fastsafetensors
        loader = SafeTensorsFileLoader(SingleGroup(), dev)
        loader.add_filenames({0: [model_path]})
        metadata = loader.meta[model_path][0].metadata
        fb = loader.copy_files_to_device()
        keys = list(fb.key_to_rank_lidx.keys())
        sd = {} # state dictionary
        for k in keys:
            sd[k] = fb.get_tensor(k)
        #fb.close() # No!
        #loader.close() # No!
        
        # Init the model to pass to ComfyUI
        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
        if len(temp_sd) > 0:
            sd = temp_sd
        model_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
        if model_config == None:
            fb.close()
            loader.close()
            raise RuntimeError("Couldn't load the model.")
        model_dtype = comfy.utils.weight_dtype(sd, "")
        model_config.set_inference_dtype(model_dtype, torch.bfloat16)
        model = model_config.get_model(sd, "", device=None)
        
        # Use this instead of load_model_weights()
        # I think 'diffusion_model' is a subclass of torch.nn.Module,
        # so we can use assign=True in load_state_dict which avoids
        # a copy. Just make sure you don't free the tensors read
        # by fastsafetensors if we use this option.
        #
        # The following duplicates the functionality of load_model_weights():
        #
        # Ensure stuff isn't duplicated on the GPU
        model = model.to(None)
        #
        sd = model_config.process_unet_state_dict(sd)
        # load_state_dict is from torch.nn.Module
        # If using assign=True, then don't free the tensors
        # loaded with fastsafetensors.
        model.diffusion_model.load_state_dict(sd, strict=False, assign=True)
        
        model = comfy.model_patcher.ModelPatcher(model, load_device=dev, offload_device=None)

        # Track in registry for the unloader
        _load_counter += 1
        _dgx_registry[model_name] = {
            "fb": fb,
            "loader": loader,
            "model_patcher": model,
            "load_id": _load_counter,
        }

        return (model,)


class DGXSparkUnloader:
    """Companion node to DGXSparkSafetensorsLoader.
    Frees all GPU/RAM memory held by a model loaded via the DGX Spark loader.
    After unloading, re-queue the workflow to reload the model automatically.

    Set 'confirm' to True and queue the workflow (or click play) to unload.
    The node is a safe no-op when confirm is False, so normal workflow runs
    won't accidentally unload your models."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "confirm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Safety toggle. Set to True to actually unload. When False the node does nothing, so normal workflow runs are safe.",
                }),
                "mode": (["selected", "all"], {
                    "default": "selected",
                    "tooltip": "'selected' unloads only the chosen model. 'all' unloads every model loaded by the DGX Spark loader.",
                }),
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "The model to unload (only used when mode is 'selected').",
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "unload_model"
    OUTPUT_NODE = True
    CATEGORY = "loaders"
    DESCRIPTION = "Frees GPU/RAM memory for models loaded by the DGX Spark Loader. Set 'confirm' to True and queue to trigger. Safe no-op when confirm is False."

    @classmethod
    def IS_CHANGED(cls, mode, model_name, confirm):
        if not confirm:
            return False
        # Always re-execute when confirm is True so the unload actually runs
        return float("nan")

    def unload_model(self, mode, model_name, confirm):
        if not confirm:
            return {"ui": {"text": ["Skipped (confirm is False)"]}}

        if mode == "all":
            names = list(_dgx_registry.keys())
            if not names:
                return {"ui": {"text": ["No models loaded via DGX Spark loader."]}}
            for name in names:
                _cleanup_model(name)
            return {"ui": {"text": [f"Unloaded {len(names)} model(s): {', '.join(names)}"]}}

        # mode == "selected"
        if model_name in _dgx_registry:
            _cleanup_model(model_name)
            return {"ui": {"text": [f"Unloaded: {model_name}"]}}
        return {"ui": {"text": [f"Not loaded via DGX Spark loader: {model_name}"]}}

