#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import math
import numpy
from collections import OrderedDict

from interface import interface, implements, method
from deepmodels import DeepFeatureRep, AttributeClassifier
from torch_models import Vgg19g, ReconModel

import torch
import torch.nn as nn
import torchvision as tv
from torch.autograd import Variable


model_urls = {
    'recon_g345': 'recon-g345-d36b8ec0.pth',
    'recon_g345_adv18': 'recon-g345-adv18-6f7ababc.pth',
    'recon_g345_d0.2': 'recon-g345-d0.2-4239ba22.pth',
}


class FitToQuantum():
  def __init__(self, quantum=64):
    self.quantum = float(quantum)

  def __call__(self, img):
    quantum = self.quantum
    size = img.size()

    if img.size(1) % int(quantum) == 0:
      pad_w = 0
    else:
      pad_w = int((quantum - img.size(1) % int(quantum)) / 2)

    if img.size(2) % int(quantum) == 0:
      pad_h = 0
    else:
      pad_h = int((quantum - img.size(2) % int(quantum)) / 2)

    res = torch.zeros(size[0],
        int(math.ceil(size[1]/quantum) * quantum),
        int(math.ceil(size[2]/quantum) * quantum))
    res[:, pad_w:(pad_w + size[1]), pad_h:(pad_h + size[2])].copy_(img)
    return res


def unfit_from_quantum(img, orig_size, quantum = 64):
  if orig_size[1] % int(quantum) == 0:
    pad_w = 0
  else:
    pad_w = int((quantum - orig_size[1] % int(quantum)) / 2)

  if orig_size[2] % int(quantum) == 0:
    pad_h = 0
  else:
    pad_h = int((quantum - orig_size[2] % int(quantum)) / 2)

  res = img[:, pad_w:(pad_w + orig_size[1]), pad_h:(pad_h + orig_size[2])].clone()
  return res


class Dataset(torch.utils.data.TensorDataset):
  def __init__(self, x, transform):
    super(Dataset, self).__init__(x, torch.zeros(x.size(0)))
    self.transform = transform

  def __getitem__(self, index):
    input = self.transform(self.data_tensor[index])
    target = self.target_tensor[index]
    return input, target


class TVLoss(nn.Module):
  def __init__(self, eps=1e-3, beta=2):
    super(TVLoss, self).__init__()
    self.eps = eps
    self.beta = beta

  def forward(self, input):
    x_diff = input[:, :, :-1, :-1] - input[:, :, :-1, 1:]
    y_diff = input[:, :, :-1, :-1] - input[:, :, 1:, :-1]

    sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, self.eps, 10000000)
    return torch.norm(sq_diff, self.beta / 2.0) ** (self.beta / 2.0)


@implements(DeepFeatureRep)
class vgg19g_torch_base(object):
  def __init__(self, **options):
    device_id = options.get('device_id', 0)
    self.forward_model = Vgg19g(pretrained = True)
    self.forward_model.eval()

    # Put it on the GPU
    self.device_id = device_id

    # Transformations for the model
    mean = torch.Tensor((0.485, 0.456, 0.406))
    stdv = torch.Tensor((0.229, 0.224, 0.225))

    self.forward_transform = tv.transforms.Compose([
      tv.transforms.Normalize(mean=mean, std=stdv),
      FitToQuantum(),
    ])
    self.reverse_transform = tv.transforms.Compose([
      tv.transforms.Normalize(mean=(-mean/stdv), std=(1/stdv)),
      tv.transforms.Lambda(lambda img: img.clamp(0, 1)),
    ])

    # Parameters
    self.tv_lambda = 10
    self.max_iter = 500

  def mean_F(self, X):
    # Storage for features
    flattened_features = None

    # Make dataloader for data
    x = torch.from_numpy(numpy.array(list(X))).permute(0, 3, 1, 2)
    loader = torch.utils.data.DataLoader(
        Dataset(x, transform = self.forward_transform),
        batch_size = 1,
        pin_memory = True,
    )

    with torch.cuda.device(self.device_id):
      self.forward_model.cuda()

      for i, (input, _) in enumerate(loader):
        #print('Image %d of %d' % (i+1, x.size(0)))
        input_var = Variable(input, volatile=True).cuda()
        feature_vars = self.forward_model(input_var)

        # Add to tally of features
        if flattened_features is None:
          flattened_features = torch.cat([f.data.sum(0).view(-1) for f in feature_vars], 0)
        else:
          flattened_features.add_(torch.cat([f.data.sum(0).view(-1) for f in feature_vars], 0))
        del input_var
        del feature_vars

      flattened_features.div_(x.size(0))

      flattened_features = flattened_features.cpu()
      self.forward_model.cpu()

    return flattened_features.numpy()

  def F_inverse(self, F, initial_image, **options):
    raise NotImplementedError()


class vgg19g_torch_recon_base(vgg19g_torch_base):
  def __init__(self, device_id, drop_rate):
    self.recon_model = ReconModel(drop_rate)
    self.recon_model.eval()
    super(vgg19g_torch_recon_base, self).__init__(device_id=device_id)

  def F_inverse(self, F, initial_image, **options):
    return self.F_inverse_with_uncertainty(F, initial_image)[0]

  def F_inverse_with_uncertainty(self, F, initial_image):
    x = torch.from_numpy(numpy.array(initial_image))
    x = x.permute(2, 0, 1)
    orig_size = x.size()

    x = self.forward_transform(x)
    x = x.contiguous().view(1, *x.size())

    with torch.cuda.device(self.device_id):
      recon_var = nn.Parameter(x.cuda(), requires_grad = True)

      # Get size of features
      self.forward_model.cuda()
      orig_feature_vars = self.forward_model(recon_var)
      self.forward_model.cpu()

      sizes = ([f.data[:1].size() for f in orig_feature_vars])
      cat_offsets = torch.cat([torch.Tensor([0]), torch.cumsum(torch.Tensor([f.data[:1].nelement() for f in orig_feature_vars]), 0)])

      # Reshape provided features to match original features
      cat_features = torch.from_numpy(F).view(-1)
      features = tuple(Variable(cat_features[int(start_i):int(end_i)].view(size)).cuda()
          for size, start_i, end_i in zip(sizes, cat_offsets[:-1], cat_offsets[1:]))

      # Do recon
      self.recon_model.cuda()
      recon_var, recon_stdv_var = self.recon_model(features, mc_samples = 20)
      recon = recon_var.data[0].cpu()
      recon_stdv = recon_stdv_var.data[0].cpu()
      self.recon_model.cpu()

    # Return the new image
    recon = self.reverse_transform(recon)
    recon = unfit_from_quantum(recon, orig_size)
    recon = recon.squeeze()
    recon = recon.permute(1, 2, 0)
    recon_stdv = unfit_from_quantum(recon_stdv, orig_size)
    recon_stdv = recon_stdv.squeeze()
    recon_stdv = recon_stdv.permute(1, 2, 0)
    return recon.numpy(), recon_stdv.numpy()


class vgg19g_recon_g345(vgg19g_torch_recon_base):
  def __init__(self, device_id):
    super(vgg19g_recon_g345, self).__init__(device_id, drop_rate = 0)
    state_dict = torch.utils.model_zoo.load_url(model_urls['recon_g345'], 'recon_models')
    self.recon_model.load_state_dict(state_dict)


class vgg19g_recon_g345_adv18(vgg19g_torch_recon_base):
  def __init__(self, device_id):
    super(vgg19g_recon_g345_adv18, self).__init__(device_id, drop_rate = 0)
    state_dict = torch.utils.model_zoo.load_url(model_urls['recon_g345_adv18'], 'recon_models')
    self.recon_model.load_state_dict(state_dict)


class vgg19g_recon_g345_d02(vgg19g_torch_recon_base):
  def __init__(self, device_id):
    super(vgg19g_recon_g345_d02, self).__init__(device_id, drop_rate = 0.2)
    state_dict = torch.utils.model_zoo.load_url(model_urls['recon_g345_d0.2'], 'recon_models')
    self.recon_model.load_state_dict(state_dict)


class vgg19g_torch(vgg19g_torch_base):
  def F_inverse_with_uncertainty(self, F, initial_image):
    res = self.F_inverse(F, initial_image)
    stdv = numpy.zeros(res.shape)
    return res, stdv

  def F_inverse(self, F, initial_image, **options):
    verbose = options.get('verbose', 0)
    x = torch.from_numpy(numpy.array(initial_image))
    x = x.permute(2, 0, 1)
    orig_size = x.size()

    x = self.forward_transform(x)
    x = x.contiguous().view(1, *x.size())

    with torch.cuda.device(self.device_id):
      self.forward_model.cuda()
      recon_var = nn.Parameter(x.cuda(), requires_grad = True)

      # Get size of features
      orig_feature_vars = self.forward_model(recon_var)
      sizes = ([f.data[:1].size() for f in orig_feature_vars])
      cat_offsets = torch.cat([torch.Tensor([0]), torch.cumsum(torch.Tensor([f.data[:1].nelement() for f in orig_feature_vars]), 0)])

      # Reshape provided features to match original features
      cat_features = torch.from_numpy(F).view(-1)
      features = tuple(Variable(cat_features[int(start_i):int(end_i)].view(size)).cuda()
          for size, start_i, end_i in zip(sizes, cat_offsets[:-1], cat_offsets[1:]))

      # Create optimizer and loss functions
      optimizer = torch.optim.LBFGS(
          params = [recon_var],
          max_iter = options['max_iter'] if 'max_iter' in options else self.max_iter,
      )
      optimizer.n_steps = 0
      criterion3 = nn.MSELoss(size_average = False).cuda()
      criterion4 = nn.MSELoss(size_average = False).cuda()
      criterion5 = nn.MSELoss(size_average = False).cuda()
      criterion_tv = TVLoss().cuda()

      # Optimize
      def step():
        self.forward_model.zero_grad()
        if recon_var.grad is not None:
          recon_var.grad.data.fill_(0)

        output_var = self.forward_model(recon_var)
        loss3 = criterion3(output_var[0], features[0])
        loss4 = criterion4(output_var[1], features[1])
        loss5 = criterion5(output_var[2], features[2])
        loss_tv = self.tv_lambda * criterion_tv(recon_var)
        loss = loss3 + loss4 + loss5 + loss_tv
        loss.backward()

        if verbose and optimizer.n_steps % 25 == 0:
          print('Step: %d  total: %.1f  conv3: %.1f  conv4: %.1f  conv5: %.1f  tv: %.3f' %
              (optimizer.n_steps, loss.data[0], loss3.data[0], loss4.data[0], loss5.data[0], loss_tv.data[0]))

        optimizer.n_steps += 1
        return loss

      optimizer.step(step)
      self.forward_model.cpu()
      recon = recon_var.data[0].cpu()

    # Return the new image
    recon = self.reverse_transform(recon)
    recon = unfit_from_quantum(recon, orig_size)
    recon = recon.squeeze()
    recon = recon.permute(1, 2, 0)
    return recon.numpy()
