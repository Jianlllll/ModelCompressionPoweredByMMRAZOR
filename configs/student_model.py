import torch
import torch.nn as nn
import torch.nn.functional as F
from  configs.teacher_model import  BasicBlock, Bottleneck, HighResolutionModule, BN_MOMENTUM


class StudentModel_HRNet_W18(nn.Module):
    def __init__(self, num_classes=1):
        super(StudentModel_HRNet_W18, self).__init__()
        
        # W18配置的通道数
        num_filters = [18, 36, 72, 144]
        
        # stem net
        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2  = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu   = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        pre_stage_channels              = [Bottleneck.expansion * 64]
        num_channels                    = [num_filters[0], num_filters[1]]
        self.transition1                = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage2, pre_stage_channels = self._make_stage(1, 2, BasicBlock, [4, 4], num_channels, num_channels)

        num_channels                    = [num_filters[0], num_filters[1], num_filters[2]]
        self.transition2                = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(4, 3, BasicBlock, [4, 4, 4], num_channels, num_channels)

        num_channels                    = [num_filters[0], num_filters[1], num_filters[2], num_filters[3]]
        self.transition3                = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(3, 4, BasicBlock, [4, 4, 4, 4], num_channels, num_channels)

        self.pre_stage_channels         = pre_stage_channels
        
        # 最后一层 - 水印解码头
        last_inp_channels = sum(num_filters)  # 18+36+72+144 = 270
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_inchannels, num_channels):
        num_branches_pre = len(num_inchannels)
        num_branches_cur = len(num_channels)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels[i] != num_inchannels[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[i], num_channels[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = [
                    nn.Sequential(
                        nn.Conv2d(num_inchannels[-1], num_channels[i], 3, 2, 1, bias=False),
                        nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    )
                ]
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_modules, num_branches, block, num_blocks, num_inchannels, num_channels, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            modules.append(
                HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        x1 = F.interpolate(y_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(y_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(y_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([y_list[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
