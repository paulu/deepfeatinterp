import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

model_urls = {
    'vgg19g': 'https://www.dropbox.com/s/cecy6wtjy97wt3d/vgg19g-4aff041b.pth?dl=1',
}

class Vgg19g(nn.Module):
  def __init__(self, pretrained=True):
    super(Vgg19g, self).__init__()
    self.features_1 = nn.Sequential(OrderedDict([
      ('conv1_1', nn.Conv2d(3, 64, kernel_size = 3, padding = 1)),
      ('relu1_1', nn.ReLU(inplace = True)),
      ('conv1_2', nn.Conv2d(64, 64, kernel_size = 3, padding = 1)),
      ('relu1_2', nn.ReLU(inplace = True)),
      ('pool1', nn.MaxPool2d(2, 2)),
      ('conv2_1', nn.Conv2d(64, 128, kernel_size = 3, padding = 1)),
      ('relu2_1', nn.ReLU(inplace = True)),
      ('conv2_2', nn.Conv2d(128, 128, kernel_size = 3, padding = 1)),
      ('relu2_2', nn.ReLU(inplace = True)),
      ('pool2', nn.MaxPool2d(2, 2)),
      ('conv3_1', nn.Conv2d(128, 256, kernel_size = 3, padding = 1)),
      ('relu3_1', nn.ReLU(inplace = True)),
    ]))
    self.features_2 = nn.Sequential(OrderedDict([
      ('conv3_2', nn.Conv2d(256, 256, kernel_size = 3, padding = 1)),
      ('relu3_2', nn.ReLU(inplace = True)),
      ('conv3_3', nn.Conv2d(256, 256, kernel_size = 3, padding = 1)),
      ('relu3_3', nn.ReLU(inplace = True)),
      ('conv3_4', nn.Conv2d(256, 256, kernel_size = 3, padding = 1)),
      ('relu3_4', nn.ReLU(inplace = True)),
      ('pool3', nn.MaxPool2d(2, 2)),
      ('conv4_1', nn.Conv2d(256, 512, kernel_size = 3, padding = 1)),
      ('relu4_1', nn.ReLU(inplace = True)),
    ]))
    self.features_3 = nn.Sequential(OrderedDict([
      ('conv4_2', nn.Conv2d(512, 512, kernel_size = 3, padding = 1)),
      ('relu4_2', nn.ReLU(inplace = True)),
      ('conv4_3', nn.Conv2d(512, 512, kernel_size = 3, padding = 1)),
      ('relu4_3', nn.ReLU(inplace = True)),
      ('conv4_4', nn.Conv2d(512, 512, kernel_size = 3, padding = 1)),
      ('relu4_4', nn.ReLU(inplace = True)),
      ('pool4', nn.MaxPool2d(2, 2)),
      ('conv5_1', nn.Conv2d(512, 512, kernel_size = 3, padding = 1)),
      ('relu5_1', nn.ReLU(inplace = True)),
    ]))

    if pretrained:
      state_dict = torch.utils.model_zoo.load_url(model_urls['vgg19g'])
      self.load_state_dict(state_dict)

  def forward(self, x):
    features_1 = self.features_1(x)
    features_2 = self.features_2(features_1)
    features_3 = self.features_3(features_2)
    return features_1, features_2, features_3


class _PoolingBlock(nn.Sequential):
  def __init__(self, n_convs, n_input_filters, n_output_filters, drop_rate):
    super(_PoolingBlock, self).__init__()
    for i in range(n_convs):
      self.add_module('conv.%d' % (i+1), nn.Conv2d(n_input_filters if i == 0 else n_output_filters, n_output_filters, kernel_size=3, padding=1))
      self.add_module('norm.%d' % (i+1), nn.BatchNorm2d(n_output_filters))
      self.add_module('relu.%d' % (i+1), nn.ReLU(inplace=True))
      if drop_rate > 0:
        self.add_module('drop.%d' % (i+1), nn.Dropout(p=drop_rate))


class _TransitionUp(nn.Sequential):
  def __init__(self, n_input_filters, n_output_filters):
    super(_TransitionUp, self).__init__()
    self.add_module('unpool.conv', nn.ConvTranspose2d(n_input_filters, n_output_filters, kernel_size=1, stride=2, output_padding=1))
    self.add_module('unpool.norm', nn.BatchNorm2d(n_output_filters))

class PatchDiscModel(nn.Sequential):
  def __init__(self, drop_rate = 0):
    super(PatchDiscModel, self).__init__()
    self.add_module('patch_conv1', nn.Conv2d(3, 64, kernel_size=3, padding=1))
    self.add_module('patch_norm1', nn.BatchNorm2d(64))
    self.add_module('patch_relu1', nn.ReLU(inplace=True))
    if drop_rate > 0:
      self.add_module('patch_drop1', nn.Dropout(p=drop_rate))
    self.add_module('patch_conv2', nn.Conv2d(64, 64, kernel_size=3, padding=1))
    self.add_module('patch_norm2', nn.BatchNorm2d(64))
    self.add_module('patch_relu2', nn.ReLU(inplace=True))
    if drop_rate > 0:
      self.add_module('patch_drop2', nn.Dropout(p=drop_rate))
    self.add_module('patch_pool', nn.MaxPool2d(2, 2))
    self.add_module('patch_conv3', nn.Conv2d(64, 64, kernel_size=3, padding=1))
    self.add_module('patch_norm3', nn.BatchNorm2d(64))
    self.add_module('patch_relu3', nn.ReLU(inplace=True))
    if drop_rate > 0:
      self.add_module('patch_drop3', nn.Dropout(p=drop_rate))
    self.add_module('patch_conv4', nn.Conv2d(64, 1, kernel_size=3, padding=1))

  def forward(self, x):
    out = super(PatchDiscModel, self).forward(x)
    return F.sigmoid(out)

class ReconModel(nn.Module):
  def __init__(self, drop_rate=0):
    super(ReconModel, self).__init__()

    self.recon5 = _PoolingBlock(3, 512, 512, drop_rate = drop_rate)
    self.upool4 = _TransitionUp(512, 512)
    self.recon4 = _PoolingBlock(3, 1024, 512, drop_rate = drop_rate)
    self.upool3 = _TransitionUp(512, 256)
    self.recon3 = _PoolingBlock(3, 512, 256, drop_rate = drop_rate)
    self.upool2 = _TransitionUp(256, 128)
    self.recon2 = _PoolingBlock(2, 128, 128, drop_rate = drop_rate)
    self.upool1 = _TransitionUp(128, 64)
    self.recon1 = _PoolingBlock(1, 64, 64, drop_rate = drop_rate)
    self.recon0 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

  def forward(self, x, mc_samples=1):
    # Non MC inference
    if mc_samples == 1 or not any([isinstance(module, nn.Dropout) for module in self.modules()]):
      res = self._forward(x)
      return res, res*0

    # MC inference
    for module in self.modules():
      if isinstance(module, nn.Dropout):
        module.train()

    means = None
    covars = None
    size = None

    for i in range(mc_samples):
      output_var = self._forward(x)
      output = output_var.data
      if size is None:
        size = output.size()

      output = output.permute(0, 2, 3, 1).contiguous().view(-1, 1, 3)
      if means is None:
        means = output.clone()
      else:
        means.add_(output)

      if covars is None:
        covars = torch.bmm(output.permute(0, 2, 1), output)
      else:
        covars.baddbmm_(output.permute(0, 2, 1), output)

    means.div_(mc_samples)
    covars.div_(mc_samples).sub_(torch.bmm(means.permute(0, 2, 1), means))

    # Set stdv to be frobenius norm
    stdvs = covars.view(-1, 9).norm(p=2, dim=1)
    stdvs.sqrt_()
    stdvs.clamp_(0, 1)

    # Reshape
    means = means.view(-1, size[2], size[3], 3).permute(0, 3, 1, 2)
    stdvs = stdvs.view(-1, 1, size[2], size[3])

    means_var = Variable(means, volatile=True)
    stdvs_var = Variable(stdvs.repeat(1, 3, 1, 1), volatile=True)
    return means_var, stdvs_var

  def _forward(self, x):
    features_1, features_2, features_3 = x

    recon5 = self.recon5(features_3)
    upool4 = self.upool4(recon5)

    recon4 = self.recon4(torch.cat([upool4, features_2], 1))
    upool3 = self.upool3(recon4)

    recon3 = self.recon3(torch.cat([upool3, features_1], 1))
    upool2 = self.upool2(recon3)

    recon2 = self.recon2(upool2)
    upool1 = self.upool1(recon2)

    recon1 = self.recon1(upool1)
    recon0 = self.recon0(recon1)

    return recon0


class PatchDiscLoss(nn.Module):
  def __init__(self):
    super(PatchDiscLoss, self).__init__()
    self.bce_loss = nn.BCELoss()

  def loss_names(self):
    return ('d-disc',)

  def forward(self, input_var, target_val):
    input_var = input_var.view(-1)
    target = torch.Tensor(input_var.data.size()).type_as(input_var.data)
    target.fill_(target_val)
    target_var = Variable(target)
    target_var.data.fill_(target_val)
    return self.bce_loss(input_var, target_var)

class ReconLoss(nn.Module):
  def __init__(self, lamda):
    super(ReconLoss, self).__init__()
    self.lamda = lamda
    self.disc_loss = PatchDiscLoss()

  def loss_names(self):
    return ('c3diff', 'c4diff', 'c5diff', 'g-disc')

  def forward(self, x, y=None):
    recon_var, conv3diff, conv4diff, conv5diff, disc_output_var = x

    return torch.cat([
      torch.norm(conv3diff)**2 / conv3diff.nelement(),
      torch.norm(conv4diff)**2 / conv4diff.nelement(),
      torch.norm(conv5diff)**2 / conv5diff.nelement(),
      self.lamda * self.disc_loss(disc_output_var, 1)
    ])
