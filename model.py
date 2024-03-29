#!/usr/bin/python -tt
"""
  Implementation of SegNet, single-task

    * Encoder network (13 conv layers - first 13 layers from VGG-16)
        1) encoderBlock
        2) convBlock
        3) maxPool
    * Convolution block (identical to encoder block, but input=output channels)
        1) Conv with filter bank
        2) Batch norm
        3) ReLU
"""
import torch.nn as nn
import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser(description='SegNet, single-task')
parser.add_argument('--checkpoint', default='checkpoint', type=str, help="Checkpoint path")
parser.add_argument('--data', default='preprocessed', type=str, help="Data path")
parser.add_argument('-B','--batch-size', default='2', type=int, help="Batch size")
parser.add_argument('-L','--learning-rate', default='1e-3', type=float, help="Learning rate")
parser.add_argument('--task', default='semantic', type=str, help="Task: semantic depth, normal")
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
parser.add_argument('--multitask', dest='multitask', action='store_true',
                    help='Multi-task (not single task)')
option = parser.parse_args()


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # Network params
        vggFilters = [64, 128, 256, 512, 512]
        self.classNum = 13


        # Initialize enc/decoder block
        self.encoderBlock = nn.ModuleList([self.subBlock([3, vggFilters[0]])])
        self.decoderBlock = nn.ModuleList([self.subBlock([vggFilters[0]])])

        # Construct rest of the blocks
        for i in range(len(vggFilters)):
            if i == 0:
                self.encoderBlock.append(self.subBlock([vggFilters[i]]))
                self.decoderBlock.insert(0, self.subBlock([vggFilters[i]]))
            elif i == 1:
                self.encoderBlock.append(self.subBlock([vggFilters[i-1], vggFilters[i]]))
                self.encoderBlock.append(self.subBlock([vggFilters[i]]))
                self.decoderBlock.insert(0, self.subBlock([vggFilters[i], vggFilters[i-1]]))
                self.decoderBlock.insert(0, self.subBlock([vggFilters[i]]))
            else:
                self.encoderBlock.append(self.subBlock([vggFilters[i-1], vggFilters[i]]))
                self.encoderBlock.append(self.subBlock([vggFilters[i]]))
                self.encoderBlock.append(self.subBlock([vggFilters[i]]))
                self.decoderBlock.insert(0, self.subBlock([vggFilters[i], vggFilters[i-1]]))
                self.decoderBlock.insert(0, self.subBlock([vggFilters[i]]))
                self.decoderBlock.insert(0, self.subBlock([vggFilters[i]]))

        # Prediction layer
        if option.task == 'semantic':
            self.predLayer = self.subBlock([vggFilters[0], self.classNum], pred=True)
        if option.task == 'depth':
            self.predLayer = self.subBlock([vggFilters[0], 1], pred=True)
        if option.task == 'normal':
            self.predLayer = self.subBlock([vggFilters[0], 3], pred=True)

        # Pooling
        self.downSample = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.upSample = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # Decoder network (13 corresponding layers)
        # Pixelwise classifcaiton layer
        # First convolution blocks in the encoder network (layers 1, 3, 5, 8, 11)

        # Initialize weights
        self.initWeights()

    # Sublock contains: conv + batch norm + ReLU
    def subBlock(self, seqFilters, pred=False):
        if len(seqFilters) == 2: # Encoder/Decoder block
            block = nn.Sequential(
            nn.Conv2d(in_channels=seqFilters[0], out_channels=seqFilters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=seqFilters[1]),
            nn.ReLU(),
            )
        else: # Conv/deconvolution block
            if pred:
                block = nn.Sequential(
                  nn.Conv2d(in_channels=seqFilters[0], out_channels=seqFilters[0], kernel_size=3, padding=1),
                  nn.Conv2d(in_channels=seqFilters[0], out_channels=seqFilters[1], kernel_size=1, padding=0),
                )
            else:
                block = nn.Sequential(
                  nn.Conv2d(in_channels=seqFilters[0], out_channels=seqFilters[0], kernel_size=3, padding=1),
                  nn.BatchNorm2d(num_features=seqFilters[0]),
                  nn.ReLU(),
                )
        return block

    def initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        indices = [0] * 5

        # Forword encoder block
        for i in range(5):
            if i < 2:
                x = self.encoderBlock[2*i+1](self.encoderBlock[2*i](x))
                x, indices[i] = self.downSample(x)
            else:
                x = self.encoderBlock[3*i-1](self.encoderBlock[3*i-2](x))
                x = self.encoderBlock[3*i](x)
                x, indices[i] = self.downSample(x)

        # Forword decoder block
        for i in range(5):
            if i < 3:
                x = self.upSample(x, indices[-i-1])
                x = self.decoderBlock[3*i+1](self.decoderBlock[3*i](x))
                x = self.decoderBlock[3*i+2](x)
            else:
                x = self.upSample(x, indices[-i-1])
                x = self.decoderBlock[2*i+4](self.decoderBlock[2*i+3](x))

        # Prediction layer
        if option.task == 'semantic':
            pred = F.log_softmax(self.predLayer(x), dim=1)
        elif option.task == 'depth':
            pred = self.predLayer(x)
        elif option.task == 'normal':
            pred = self.predLayer(x)
            pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)

        return pred
