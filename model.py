import torch
import torch.nn as nn

class block(nn.Module):     # 使用bottleneck结构，这里输出始终是mid_channels*4，
                            # 而且feature map的大小不变
    def __init__(self, in_channels, mid_channels, 
                            identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1,
                                stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, 
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x.clone()    # 将输入复制下来

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # 如果有必要下采样，则进行加采样，否则直接短接。
        # 下采样一般会在大小不一致的时候使用
        # 此方法是indentity shortcut，
        if self.identity_downsample is not None:    
            identity = self.identity_downsample(identity)
        
        x += identity       # 进行短接
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, 
                                stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], mid_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], mid_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], mid_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], mid_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)   # 因为残差块使用bottleneck，因此大小扩张了4倍

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)       # 将卷积输出展平
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, mid_channels, stride):
        identity_downsample = None
        layers = []

        # 判断如果步长为2，则feature map的大小肯定变了；
        # 或者经过了bottleneck后，深度变4倍了，则进行
        # 下采样操作，具体就是使用一个1*1的卷积网络进行匹配
        if stride != 1 or self.in_channels != mid_channels*4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, mid_channels*4, kernel_size=1, 
                            stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * 4)
            )
        
        layers.append(
            block(self.in_channels, mid_channels, identity_downsample, stride)
        )

        self.in_channels = mid_channels * 4
        
        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, mid_channels))
        
        return nn.Sequential(*layers)
    
def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channels, num_classes)

def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)

def test():
    net = ResNet101(img_channels=3, num_classes=1000)
    y = net(torch.randn(4, 3, 100, 100)).to("cuda")
    print(y.size())

if __name__ == "__main__":
    test()

