from abc import ABCMeta, abstractmethod
import omegaconf
from omegaconf import OmegaConf
from torch import nn
from copy import copy

class MetaModel(ABCMeta):
    def __prepare__(name, bases, **kwds):
        total_conf = OmegaConf.create()
        for base in bases:
            for key in ('base_default_conf', 'default_conf'):
                update = getattr(base, key, {})
                if isinstance(update, dict):
                    update = OmegaConf.create(update)
                total_conf = OmegaConf.merge(total_conf, update)
        return dict(base_default_conf = total_conf)
class BaseModel(nn.Module, metaclass=MetaModel):
    default_conf = {
        'name': None,
        'trainable': True,
        'freeze_batch_normalization': False,
    }
    required_data_keys = []
    strict_conf = True

    def __init__(self, conf):
        super().__init__()
        default_conf = OmegaConf.merge(self.base_default_conf, OmegaConf.create(self.default_conf))
        if self.strict_conf:
            OmegaConf.set_struct(default_conf, True)

        if 'pad' in conf and 'pad' not in default_conf:
            with omegaconf.read_write(conf):
                with omegaconf.open_dict(conf):
                    conf['interpolation'] = {'pad': conf.pop('pad')}

        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)

        self.conf = conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)
        self.required_data_keys = copy(self.required_data_keys)
        self._init(conf)

        if not conf.trainable:
            for p in self.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        if self.conf.freeze_batch_normalization:
            self.apply(freeze_bn)

        return self

    def forward(self, data):
        def recursive_key_check(expected, given):
            for key in expected:
                assert key in given, f'Missing key {key} in data'
                if isinstance(expected, dict):
                    recursive_key_check(expected[key], given[key])

        recursive_key_check(self.required_data_keys, data)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def metrics(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError

