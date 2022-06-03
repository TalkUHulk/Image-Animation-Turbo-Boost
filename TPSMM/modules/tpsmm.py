from modules.keypoint_detector import KPDetector
from modules.bg_motion_predictor import BGMotionPredictor
from modules.dense_motion_inpainting import DenseMotionInpaintingNetwork
import torch.nn as nn


class TPSMM(nn.Module):
    def __init__(self, config):
        super(TPSMM, self).__init__()

        self.kp_detector = KPDetector(10)
        self.dense_motion_inpainting_network = DenseMotionInpaintingNetwork(**config['model_params']['common_params'],
                                                                            **config['model_params'][
                                                                                'dense_motion_params'])

        self.bg_predictor = BGMotionPredictor()

    def forward(self, kp_source, source_image, driving_image):
        kp_driving = self.kp_detector(driving_image)
        bg_params = self.bg_predictor(source_image, driving_image)

        out = self.dense_motion_inpainting_network(source_input=source_image, kp_driving=kp_driving,
                                                   kp_source=kp_source, bg_param=bg_params)
        return out
