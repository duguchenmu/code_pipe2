import numpy
import torch
from .base_model import BaseModel
from .. import get_model

class TwoViewPipePoint(BaseModel):
    default_conf = {
        'extractor': {
            'name': 'superpoint',
            'trainable': False,
        },
        'use_points': True,
        'randomize_num_kp': False,
        'detector': {'name': None},
        'descriptor': {'name': None}
    }

    required_data_keys = ['image0', 'image1']
    strict_conf = False
    components = ['extractor', 'detector', 'descriptor']

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(conf.extractor)
        else:
            if self.conf.detector.name:
                self.detector = get_model(conf.detector.name)(conf.detector)
            else:
                self.required_data_keys += ['keypoints0', 'keypoints1']
            if self.conf.descriptor.name:
                self.descriptor = get_model(conf.descriptor.name)(conf.descriptor)
            else:
                self.required_data_keys += ['descriptors0', 'descriptors1']

    def _forward(self, data):

        def process_siamese(data, i):
            data_i = {k[:-1]: v for k, v in data.items() if k[-1] == i}
            if self.conf.extractor.name:
                pred_i = self.extractor(data_i)
            else:
                pred_i = {}
                if self.conf.detector.name:
                    pred_i = self.detector(data_i)
                else:
                    for k in ['keypoints', 'keypoint_scores', 'descriptors']:
                        if k in data_i:
                            pred_i[k] = data_i[k]
                if self.conf.descriptor.name:
                    pred_i = {**pred_i, **self.descriptor({**data_i, **pred_i})}
            return pred_i

        pred0 = process_siamese(data, '0')
        pred1 = process_siamese(data, '1')

        pred = {**{k + '0': v for k, v in pred0.items()},
                **{k + '1': v for k, v in pred1.items()}}

        return pred

    def loss(self, pred, data):
        losses = {}
        total = 0
        for k in self.components:
            if self.conf[k].name:
                try:
                    losses_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                total = losses_['total'] + total
        return {**losses, 'total': total}

    def metrics(self, pred, data):
        metrics = {}
        for k in self.components:
            if self.conf[k].name:
                try:
                    metrics_ = getattr(self, k).metrics(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                metrics = {**metrics, **metrics_}
        return metrics