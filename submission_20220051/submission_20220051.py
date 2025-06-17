import torch
import torch.nn as nn
import torch.nn.functional as F

def Split(x, p):
    c = int(x.size()[1])
    c1 = round(c * (1 - p))
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = GNSELU(nOut)
    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output
    
class GNSELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        num_groups = 32 if nIn >= 32 else nIn
        self.norm = nn.GroupNorm(num_groups, nIn)
        self.acti = nn.SELU(inplace=True)

    def forward(self, x):
        return self.acti(self.norm(x))


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        nConv = nOut - nIn if self.nIn < self.nOut else nOut
        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.bn_prelu = GNSELU(nOut)
        
    def forward(self, input):
        output = self.conv3x3(input)
        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)
        output = self.bn_prelu(output)
        return output

class TCA(nn.Module):
    def __init__(self, c, d=1, kSize=3, dkSize=3):
        super().__init__()
        self.conv3x3 = Conv(c, c, kSize, 1, padding=1, bn_acti=True)
        self.dconv3x3 = Conv(c, c, (dkSize, dkSize), 1,
                             padding=(1, 1), groups=c, bn_acti=True)
        self.ddconv3x3 = Conv(c, c, (dkSize, dkSize), 1,
                              padding=(1 * d, 1 * d), groups=c, dilation=(d, d), bn_acti=True)
        self.bp = GNSELU(c)
    def forward(self, input):
        br = self.conv3x3(input)
        br1 = self.dconv3x3(br)
        br2 = self.ddconv3x3(br)
        br = br + br1 + br2
        output = self.bp(br)
        return output

class ResidualPCT(nn.Module):
    def __init__(self, nIn, d=1, p=0.5):
        super().__init__()
        self.p = p
        c = int(nIn) - round(int(nIn) * (1 - p))
        self.TCA = TCA(c, d)
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=True)
        
    def forward(self, input):
        output1, output2 = Split(input, self.p)
        output2 = self.TCA(output2)
        output = torch.cat([output1, output2], dim=1)
        output = self.conv1x1(output)
        return output + input

class BoundaryAwareDAD(nn.Module):
    def __init__(self, c2, c1, classes):
        super().__init__()
        self.c1 = c1
        self.conv1x1_c = Conv(c2, c1, 1, 1, padding=0, bn_acti=True)
        self.conv1x1_neg = Conv(c1, c1, 1, 1, padding=0, bn_acti=True)
        self.conv3x3 = Conv(c1, c1, (3, 3), 1, padding=(1, 1), groups=c1, bn_acti=True)
        self.conv1x1 = Conv(c1, classes, 1, 1, padding=0, bn_acti=False)
        self.sobel = nn.Conv2d(c1, 2 * c1, kernel_size=3, padding=1, groups=c1, bias=False)
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel = torch.cat([sobel_kernel_x, sobel_kernel_y], dim=0)
        full_sobel_kernel = sobel_kernel.repeat(c1, 1, 1, 1)
        self.sobel.weight.data = full_sobel_kernel
        self.sobel.weight.requires_grad = False
        
    def forward(self, X, Y):
        X_map = torch.sigmoid(X)
        F_sg = X_map
        Yc = self.conv1x1_c(Y)
        Yc_map = torch.sigmoid(Yc)
        Neg_map = self.conv1x1_neg(-Yc_map)
        F_rg = Neg_map * Yc_map + Yc
        F_rg = F.interpolate(F_rg, F_sg.size()[2:], mode='bilinear', align_corners=False)
        combined_features = F_sg * F_rg
        combined_features = self.conv3x3(combined_features)
        edge_gradients = self.sobel(combined_features)
        gx = edge_gradients[:, :self.c1, :, :]
        gy = edge_gradients[:, self.c1:, :, :]
        gradient_magnitude = torch.sqrt(gx**2 + gy**2)
        edge_map = torch.sigmoid(gradient_magnitude)
        attention_features = combined_features * (1 + edge_map)
        output = self.conv1x1(attention_features)
        return output

class submission_20220051(nn.Module):
    def __init__(self, in_channels, num_classes, block_1=3, block_2=6, C=4, P=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.Init_Block = nn.Sequential(
            Conv(in_channels, C, 3, 2, padding=1, bn_acti=True),
            Conv(C, C, 3, 1, padding=1, bn_acti=True),
            Conv(C, C, 3, 1, padding=1, bn_acti=True)
        )
        dilation_block_1 = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        dilation_block_2 = [4, 4, 8, 8, 16, 16, 32, 32, 32, 32, 32, 32]
        self.LC_Block_1 = nn.Sequential()
        self.LC_Block_1.add_module("downsample", DownSamplingBlock(C, C * 2))
        for i in range(block_1):
            self.LC_Block_1.add_module("LC_Module_1_" + str(i), ResidualPCT(nIn=C * 2, d=dilation_block_1[i], p=P))
        self.LC_Block_2 = nn.Sequential()
        self.LC_Block_2.add_module("downsample", DownSamplingBlock(C * 2, C * 4))
        for i in range(block_2):
            self.LC_Block_2.add_module("LC_Module_2_" + str(i), ResidualPCT(nIn=C * 4, d=dilation_block_2[i], p=P))
        self.DAD = BoundaryAwareDAD(C * 4, C * 2, num_classes)
        
    def forward(self, input):
        output0 = self.Init_Block(input)
        output1 = self.LC_Block_1(output0)
        output2 = self.LC_Block_2(output1)
        out = self.DAD(output1, output2)
        out = F.interpolate(out, size=input.size()[2:], mode='bilinear', align_corners=False)
        return out