# src/agent_eval_framework/utils/class_loader.py
import importlib
from typing import Any

def load_class(class_path: str) -> Any:
    try:
        module_path, class_name = class_path.rsplit('.', 1)
    except ValueError:
        raise ImportError(f"Invalid class path format: {class_path}. Expected 'module.Class'.")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}")
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}'.")
