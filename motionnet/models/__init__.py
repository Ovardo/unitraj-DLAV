from motionnet.models.ptr.ptr import PTR
from motionnet.models.simpl.simpl import SIMPL

__all__ = {
    'ptr': PTR,
    'simple': SIMPL,
}


def build_model(config):

    model = __all__[config.method.model_name](
        config=config
    )

    return model
