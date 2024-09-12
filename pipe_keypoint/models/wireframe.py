import numpy as np
import torch
from .base_model import BaseModel
from .superpoint import SuperPoint, sample_descriptors
# from sklearn.cluster import

class SPWireframeDescriptor(BaseModel):
    default_conf = {
        'sp_params': {
            'has_detector': True,
            'has_descriptor': True,
            'descriptor_dim': 256,
            'trainable': False,

            # Inference
            'return_all': True,
            'sparse_outputs': True,
            'nms_radius': 4,
            'detection_threshold': 0.005,
            'max_num_keypoints': 1000,
            'force_num_keypoints': True,
            'remove_borders': 4,
        },
        'wireframe_params': {
            'merge_points': True,
            'merge_line_endpoints': True,
            'nms_radius': 3,
            'max_n_junctions': 500,
        },
        'max_n_lines': 250,
        'min_length': 15,
    }
    required_data_keys = ['image']

    def _init(self, conf):
        self.conf = conf
        self.sp = SuperPoint(conf.sp_params)



    def _forward(self, data):
        b_size, _, h, w = data['image'].shape
        device = data['image'].device

        if not self.conf.sp_params.force_num_keypoints:
            assert b_size == 1, "Only batch size of 1 accepted for non padded inputs"

        pred = self.sp(data)


        return pred

    @staticmethod
    def endpoints_pooling(segs, all_descriptors, img_shape):
        assert segs.ndim == 4 and segs.shape[-2:] == (2, 2)
        filter_shape = all_descriptors.shape[-2:]
        scale_x = filter_shape[1] / img_shape[1]
        scale_y = filter_shape[0] / img_shape[0]

        scaled_segs = torch.round(segs * torch.tensor([scale_x, scale_y]).to(segs)).long()
        scaled_segs[..., 0] = torch.clip(scaled_segs[..., 0], 0, filter_shape[1] - 1)
        scaled_segs[..., 1] = torch.clip(scaled_segs[..., 1], 0, filter_shape[0] - 1)
        line_descriptors = [all_descriptors[None, b, ..., torch.squeeze(b_segs[..., 1]), torch.squeeze(b_segs[..., 0])]
                            for b, b_segs in enumerate(scaled_segs)]
        line_descriptors = torch.cat(line_descriptors)
        return line_descriptors  # Shape (1, 256, 308, 2)

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        return {}
