import torch
#import tensorflow as tf

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        return torch.dist(gt, pred, p=2) #0# tf.norm(B - A, axis=-1) #tf.norm(gt - pred, axis=-1)