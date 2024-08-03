import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from attention import se_block,CBAM,eca_block,ChannelAttention,SpatialAttention,StripPooling,EMA
atteionb=[se_block,CBAM,eca_block,ChannelAttention,SpatialAttention,StripPooling,EMA]
class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)  # 使用了切片操作[:4]，它从模型的features部分选择了前四个子模块，并将输入x传递给它们。这些子模块可能是提取输入图像中低级别特征的部分。
        the_three_features = self.features[:7](x)  #72*72*32
        the_four_features = self.features[:11](x)  # 36*36*64
        x = self.features[4:](low_level_features)  # 模型的剩余部分（从第五个子模块开始），并将结果保存在变量x中。这个部分可能包含更深层次的特征提取和语义分割任务所需的其他操作
        return low_level_features, the_three_features, the_four_features, x

    # -----------------------------------------#






#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1,cbam_ratio=16):
        super(ASPP, self).__init__()
        self.in_channels = dim_in
        self.branch1 = nn.Sequential(
            #CondConv2D(dim_in,dim_out,1,1,padding=0,dilation=rate,groups=1,bias=True,
                       #padding_mode='zeros',num_experts=3,dropout_rate=0.2),
            #DOConv2d(dim_in,dim_out,1,None,1,0,dilation=1,groups=1,bias=True,
                     #padding_mode='zeros'),
            #ODConv2d(dim_in,dim_out,1,1,padding=0,dilation=rate,groups=1,reduction=0.0625,
                     #kernel_num=4),
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            eca_block(dim_out,1,2)
            #ChannelAttention(dim_out, ratio=cbam_ratio)
            #CBAM(dim_out, ratio=cbam_ratio)
        )
        self.branch2 = nn.Sequential(
            #CondConv2D(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, groups=1, bias=True,
                       #padding_mode='zeros', num_experts=3, dropout_rate=0.2),
            #DOConv2d(dim_in, dim_out, 3, None, 1, padding=6*rate, dilation=6*rate, groups=1,
                     #bias=True, padding_mode='zeros'),
            #ODConv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, groups=1, reduction=0.0625,
                     #kernel_num=4),
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            eca_block(dim_out, 1, 2)
            #ChannelAttention(dim_out, ratio=cbam_ratio)
            #CBAM(dim_out, ratio=cbam_ratio)
        )
        self.branch3 = nn.Sequential(
            #CondConv2D(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, groups=1, bias=True,
                       #padding_mode='zeros', num_experts=3, dropout_rate=0.2),
            #DOConv2d(dim_in, dim_out, 3, None, 1, padding=12 * rate, dilation=12 * rate, groups=1,
                     #bias=True, padding_mode='zeros'),
            #ODConv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, groups=1, reduction=0.0625,
                     #kernel_num=4),
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            eca_block(dim_out, 1, 2)
            #ChannelAttention(dim_out, ratio=cbam_ratio)
            #CBAM(dim_out, ratio=cbam_ratio)
        )
        self.branch4 = nn.Sequential(
            #CondConv2D(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, groups=1, bias=True,
                       #padding_mode='zeros', num_experts=3, dropout_rate=0.2),
            #DOConv2d(dim_in, dim_out, 3, None, 1, padding=18 * rate, dilation=18 * rate, groups=1,
                     #bias=True, padding_mode='zeros'),
            #ODConv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, groups=1, reduction=0.0625,
                     #kernel_num=4),
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            eca_block(dim_out, 1, 2)
            #ChannelAttention(dim_out, ratio=cbam_ratio)
            #CBAM(dim_out, ratio=cbam_ratio)
        )
        #self.branch5_conv = ODConv2d(dim_in, dim_out, 1, 1, padding=0, dilation=1, groups=1, reduction=0.0625,
                     #kernel_num=4)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        #self.ema = EMA(channels=1280,factor=8)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            eca_block(dim_out, 1, 2)
            #ChannelAttention(dim_out, ratio=cbam_ratio)
            #CBAM(dim_out, ratio=cbam_ratio)#dim_out * 5
        )
        #self.cam = ChannelAttention(in_planes=256,reduction=1)
        self.sam = SpatialAttention(in_channels=1280,kernel_size=7)
        #self.SP = StripPooling(320, up_kwargs={'mode': 'bilinear', 'align_corners': True})
        #self.eca2 = ECANet(in_channels=320,gamma=2,b=1)

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x) #[2,256,32,32]
        #conv1x1 = self.cam(conv1x1)
        conv3x3_1 = self.branch2(x)#[2,256,32,32]
        #conv3x3_1 = self.cam(conv3x3_1)
        conv3x3_2 = self.branch3(x)#[2,256,32,32]
        #conv3x3_2 = self.cam(conv3x3_2)
        conv3x3_3 = self.branch4(x)#[2,256,32,32]
        #conv3x3_3 = self.cam(conv3x3_3)
        #sp = self.SP(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)#[2,256,32,32]
        #global_feature = self.cam(global_feature)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3,global_feature], dim=1)
        #feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1) #[2,1280,32,32]
        #feature_cat = self.sam(feature_cat)
        #print(feature_cat.shape)
        #feature_cat = self.ema(feature_cat)
        feature_cat = self.sam(feature_cat)
        result = self.conv_cat(feature_cat)
        return result








class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320#320
            low_level_channels = 24
            the_three_channels = 32
            the_four_channels = 64
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        #self.CA = CoordAtt(320, 320)
        self.cbam_the_three_channels = CBAM(32,16,7)
        self.cbam_the_four_channels = CBAM(64, 16, 7)
        self.cbam_the_low_level_channels = CBAM(24,16,7)
        #self.CA_the_three_channels = CoordAtt(inp=32,oup=32)#单独3通道CA73.6
        #self.CA_the_four_channels = CoordAtt(inp=64, oup=64, reduction=32)#单独4通道CA不行
        #self.CA_low_level_channels = CoordAtt(inp=24, oup=24, reduction=32)
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
        # self.scSE = scSE(in_ch=256)
        #self.eca = eca_block(channel=48,gamma=2,b=1) #√用于低层次
        # self.shuffleAttention = ShuffleAttention(channel=256,reduction=16,G=8)
        # self.sc=scSE(in_ch=256)
        # self.CA=CoordAtt(inp=48,oup=48,reduction=32)
        #self.SE = se_block(channel=256,ratio=4) #√
        # self.EMA = EMA() √
        # self.Channel_Att = Channel_Att(channels=256)
        # self.Att =Att(channels=256,shape=[8,256,32,32],out_channels=None,no_spatial=True)
        # self.Channel_Att1 = Channel_Att(channels=24)
        # self.Att1 = Att(channels=24,shape=[8,24,128,128],out_channels=None,no_spatial=True)
        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels+the_three_channels+the_four_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1) #kernel_size=1
        #---------------------------CFF---------------------------------------------
    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, the_three_features, the_four_features,x = self.backbone(x)
        #print(the_four_features.size())
        #x = self.CA(x)
        the_three_features = self.cbam_the_three_channels(the_three_features)
        the_four_features = self.cbam_the_four_channels(the_four_features)
        low_level_features = self.cbam_the_low_level_channels(low_level_features)
        #the_three_features=self.CA_the_three_channels(the_three_features)
        #the_four_features=self.CA_the_four_channels(the_four_features)
        #low_level_features=self.CA_low_level_channels(low_level_features)
        x = self.aspp(x)
        the_three_features_up = F.interpolate(the_three_features,
                                              size=(low_level_features.size(2), low_level_features.size(3)),
                                              mode='bilinear', align_corners=True)
        the_four_features_up = F.interpolate(the_four_features,
                                             size=(low_level_features.size(2), low_level_features.size(3)),
                                             mode='bilinear', align_corners=True)
        low_level_features = self.shortcut_conv(
            torch.cat((low_level_features, the_three_features_up, the_four_features_up), dim=1))
        # x = self.scSE(x)
        # x = self.shuffleAttention(x)
        # x = self.ela(x)
        # x = self.ms_cam(x)
        # x = self.scSE(x)
        # x = self.ema(x) √
        # x = self.SE(x)
        # x=self.Channel_Att(x)
        # x=self.Att(x)
        # low_level_features = self.Channel_Att1(low_level_features)
        # low_level_features = self.Att1(low_level_features)
        #low_level_features = self.shortcut_conv(low_level_features)
        #F1 = self.F1(the_three_features)  # 72*72*32-72*72*192
        # 36*36*64-72*72*64
        #low_level_features = self.eca(low_level_features) #√用于低层次特征
        # low_level_features = self.CA(low_level_features)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        #x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          #align_corners=True)
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        #x = self.cat_conv(torch.cat((low_level_features, F2_1), dim=1))  # 144*144*304-144*144*256
        #x = self.SE(x)
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
import torch
import torch.nn as nn
import math
class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output