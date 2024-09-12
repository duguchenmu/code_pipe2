import collections
from pathlib import Path
import torch

PIPE_KEYPOINT_ROOT = Path(__file__).parent.parent

def get_class(mod_name, base_path, BaseClass):
    import inspect
    mod_path = '{}.{}'.format(base_path, mod_name)
    mod = __import__(mod_path, fromlist=[''])
    classes = inspect.getmembers(mod, inspect.isclass)
    classes = [c for c in classes if c[1].__module__ == mod_path]
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]

def get_model(name):
    from .models.base_model import BaseModel
    return get_class('models.' + name, __name__, BaseModel)

def numpy_image_to_torch(image):
    if image.ndim ==3:
        image = image.transpose((2, 0, 1))
    elif image.ndim ==2:
        image = image[None]
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy(image / 255.).float()

def map_tensor(input_, func):
    if isinstance(input_, (str, bytes)):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    else:
        return func(input_)

def batch_to_np(batch):
    return map_tensor(batch, lambda t: t.detach().cpu().numpy()[0])