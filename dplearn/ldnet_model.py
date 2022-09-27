import torch
import torch.nn as nn
from dplearn.hourglass_model import Conv, Hourglass, Residual
import cv2 as cv
import numpy as np


class Convert(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convert, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        return self.conv(x)


class LDNet(nn.Module):

    def __init__(self, nstack=2, layer=6, in_channel=265, out_channel=55, increase=0):
        super(LDNet, self).__init__()
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            nn.MaxPool2d(2, 2),
            Residual(128, 128),
            Residual(128, in_channel)
        )
        self.hourglass = nn.ModuleList([nn.Sequential(Hourglass(layer, in_channel, inc=increase)) for _ in range(nstack)])
        self.feature = nn.ModuleList([nn.Sequential(Residual(in_channel, in_channel), Conv(in_channel, in_channel, 1, bn=True, relu=True)) for _ in range(nstack)])
        self.outs = nn.ModuleList([Conv(in_channel, out_channel, 1, bn=False, relu=False) for _ in range(nstack)])
        self.merge_feature = nn.ModuleList([Convert(in_channel, in_channel) for _ in range(nstack - 1)])
        self.merge_pred = nn.ModuleList([Convert(out_channel, in_channel) for _ in range(nstack - 1)])

    def forward(self, x):
        x = self.pre(x)
        #print(x.shape)
        #cv.imshow('input', x[0].data.numpy())
        #cv.waitKey(0)
        #cv.imwrite("input_feature.png", x)
        heat_maps = []
        for i in range(self.nstack):
            hg = self.hourglass[i](x)
            feature = self.feature[i](hg)
            #print(feature)
            #print(feature.shape)
            #cv.imshow("feature ", feature)
            #cv.imwrite("feature_"+str(i)+".png", feature)
            
            pred = self.outs[i](feature)
            #print(pred.shape)
            #cv.imshow("prediction ", pred)
            #cv.imwrite("prediction_"+str(i)+".png", (pred[0].data.numpy()*255).astype(np.uint8))
            heat_maps.append(pred)
            if i < self.nstack - 1:
                x = x + self.merge_pred[i](pred) + self.merge_feature[i](feature)
                #cv.imshow("new input", x)
                #cv.imwrite("new_input_"+str(i)+".png", x)
        return pred


if __name__ == '__main__':
    model = LDNet(4, 2, 256, 55)
    x = torch.zeros((1, 3, 256, 256))
    out = model(x)
    print(out)
    for i in range(len(out)):
        print(out[i].shape)
