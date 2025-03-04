import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_kernel(kernel_len=16, nsig=10):  # nsig 标准差 ，kernlen=16核尺寸
    GaussianKernel = cv2.getGaussianKernel(kernel_len, nsig) \
        * cv2.getGaussianKernel(kernel_len, nsig).T
    return GaussianKernel


class Gaussian_kernel(nn.Module):
    def __init__(self,
                 # device,
                 kernel_len, nsig=20):
        super(Gaussian_kernel, self).__init__()
        self.kernel_len = kernel_len
        kernel = get_kernel(kernel_len=kernel_len, nsig=nsig)  # 获得高斯卷积核
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展两个维度
        # self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        self.padding = torch.nn.ReplicationPad2d(int(self.kernel_len/2))

    def forward(self, x):  # x1是用来计算attention的，x2是用来计算的Cs
        x = self.padding(x)
        # 对三个channel分别做卷积
        res = []
        for i in range(x.shape[1]):
            res.append(F.conv2d(x[:, i:i+1], self.weight))
        x_output = torch.cat(res, dim=1)
        return x_output


def hfc_mul_mask(hfc_filter, image, mask):
    hfc = hfc_filter(image, mask)
    # return hfc
    return (hfc + 1) * mask - 1


class HFCFilter(nn.Module):
    def __init__(self,
                 filter_width=27, nsig=9, ratio=4, sub_low_ratio=1, sub_mask=True, is_clamp=True):
        super(HFCFilter, self).__init__()
        self.gaussian_filter = Gaussian_kernel(
            # device,
            filter_width, nsig=nsig)
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)
        self.max = 1.0
        self.min = -1.0
        self.ratio = ratio
        self.sub_low_ratio = sub_low_ratio
        self.sub_mask = sub_mask
        self.is_clamp = is_clamp

    def median_padding(self, x, mask):
        m_list = []
        batch_size = x.shape[0]
        for i in range(x.shape[1]):
            m_list.append(x[:, i].view([batch_size, -1]
                                       ).median(dim=1).values.view(batch_size, -1) + 0.2)
        median_tensor = torch.cat(m_list, dim=1)
        median_tensor = median_tensor.unsqueeze(2).unsqueeze(2)
        mask_x = mask * x
        padding = (1 - mask) * median_tensor
        return padding + mask_x

    def forward(self, x, mask):
        assert mask is not None
        x = self.median_padding(x, mask)
        gaussian_output = self.gaussian_filter(x)
        res = self.ratio * (x - self.sub_low_ratio * gaussian_output)
        if self.is_clamp:
            res = torch.clamp(res, self.min, self.max)
        if self.sub_mask:
            res = (res + 1) * mask - 1

        return res
