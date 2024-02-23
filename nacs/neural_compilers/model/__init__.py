from neural_compilers.model.nc.core import NeuralCompiler
from neural_compilers.model.external.perceiver_io import PerceiverIO


def __getattr__(name):
    if name in globals():
        return globals()[name]
    # If we're here, name is not in globals
    # Look for it in timm
    import timm

    model_cls = getattr(timm.models, name, None)
    if model_cls is None:
        raise AttributeError(f"Model {name} not found.")
    return model_cls
