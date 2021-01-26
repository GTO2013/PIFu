from ..net_util import *

class SimpleEncoder(nn.Module):
    def __init__(self, opt):
        super(SimpleEncoder, self).__init__()
        self.num_modules = opt.num_stack

        self.opt = opt
        groupNormSize = 256

        self.conv1 = nn.Conv2d(3 if self.opt.use_normal_input else 1, 64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
       # self.bn1 = nn.GroupNorm(32, 64)
        self.conv2 = ConvBlock(64, 128, self.opt.norm)
        self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.conv3 = ConvBlock(128, 128, self.opt.norm)
        self.down_conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.conv4 = ConvBlock(128, 128, self.opt.norm)
        self.down_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.conv5 = ConvBlock(128, 128, self.opt.norm)
        self.down_conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv6 = ConvBlock(128, 128, self.opt.norm)

    def forward(self, x):
        outputs = []
        x = F.relu(self.bn1(self.conv1(x)), True)
        outputs.append(x)
        x = self.conv2(x)
        x = self.down_conv2(x)
        outputs.append(x)
        x = self.conv3(x)
        x = self.down_conv3(x)
        outputs.append(x)
        x = self.conv4(x)
        x = self.down_conv4(x)
        outputs.append(x)

        return outputs
