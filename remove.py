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
from util import parseTag

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # Network params
        vggFilters = [64, 128, 256, 512, 512]
        self.classNum = 13


        # Initialize enc/decoder block
        self.encoderBlock = nn.ModuleList([self.subBlock([3, vggFilters[0]])])
        self.decoderBlock = nn.ModuleList([self.subBlock([vggFilters[-1]])])

        # Construct rest of the blocks
        for i in range(len(vggFilters)):
            if i == 0:
                self.encoderBlock.append(self.subBlock([vggFilters[i]]))
                self.decoderBlock.insert(0, self.subBlock([vggFilters[-i-1]]))
            elif i == 1:
                self.encoderBlock.append(self.subBlock([vggFilters[i-1], vggFilters[i]]))
                self.encoderBlock.append(self.subBlock([vggFilters[i]]))
                self.decoderBlock.insert(0, self.subBlock([vggFilters[-i-1], vggFilters[-i]]))
                self.decoderBlock.insert(0, self.subBlock([vggFilters[-i-1]]))
            else:
                self.encoderBlock.append(self.subBlock([vggFilters[i-1], vggFilters[i]]))
                self.encoderBlock.append(self.subBlock([vggFilters[i]]))
                self.encoderBlock.append(self.subBlock([vggFilters[i]]))
                self.decoderBlock.insert(0, self.subBlock([vggFilters[-i-1], vggFilters[-i]]))
                self.decoderBlock.insert(0, self.subBlock([vggFilters[-i-1]]))
                self.decoderBlock.insert(0, self.subBlock([vggFilters[-i-1]]))

        # Prediction layer
        if option.task == 'semantic':
            self.predLayer = self.subBlock([filter[0], self.classNum], pred=True)
        if option.task == 'depth':
            self.predLayer = self.subBlock([filter[0], 1], pred=True)
        if option.task == 'normal':
            self.predLayer = self.subBlock([filter[0], 3], pred=True)

        # Pooling
        self.downSample = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.upSample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder network (13 corresponding layers)
        # Pixelwise classifcaiton layer
        # First convolution blocks in the encoder network (layers 1, 3, 5, 8, 11)

        # Initialize weights
        initWeights()

    # Sublock contains: conv + batch norm + ReLU
    def subBlock(self, Filter, pred=False):
        if len(Filter) == 2: # Encoder/Decoder block
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
                nn.BatchNorm2d(num_features=seqFilters[1]),
                nn.ReLU(),
                )
        return block

    def initWeights(self):
        for m in self.Modules():
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
    			x = self.encoderBlock[3*i+1](self.encoderBlock[3*i](x))
    			x = self.encoderBlock[3*i+2](x)
    			x = self.upSample(x)
    		else:
    			x = self.encoderBlock[2*i+4](self.encoderBlock[2*i+3](x))
    			x = self.upSample(x)

        if option.task == 'semantic':
            pred = F.log_softmax(self.predLayer(x), dim=1)
