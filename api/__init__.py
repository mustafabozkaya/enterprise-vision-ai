"""
API Package
FastAPI uygulaması için API rotaları ve şemaları
"""

# Lazy imports to avoid circular imports
def __getattr__(name):
    if name == "inference":
        from api.routes import inference
        return inference
    elif name == "upload":
        from api.routes import upload
        return upload
    raise AttributeError(f"module 'api' has no attribute '{name}'")

__all__ = ["inference", "upload"]
