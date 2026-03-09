"""
Services Package
İş mantığı servisleri
"""

# Lazy imports to avoid circular imports
def __getattr__(name):
    if name == "inference_service":
        from services import inference_service
        return inference_service
    elif name == "model_service":
        from services import model_service
        return model_service
    elif name == "preprocessing":
        from services import preprocessing
        return preprocessing
    elif name == "utils":
        from services import utils
        return utils
    raise AttributeError(f"module 'services' has no attribute '{name}'")

__all__ = ["inference_service", "model_service", "preprocessing", "utils"]
