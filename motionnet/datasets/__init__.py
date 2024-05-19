from .ptr_dataset import PTRDataset

__all__ = {
    'ptr': PTRDataset,
    'simpl' SIMPLDataset,
}

def build_dataset(config,val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
