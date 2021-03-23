# Copyright 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn.functional as F

# LeCam Regularziation loss
def lecam_reg(dis_real, dis_fake, ema):
  reg = torch.mean(F.relu(dis_real - ema.D_fake).pow(2)) + \
        torch.mean(F.relu(ema.D_real - dis_fake).pow(2))
  return reg

# ------ non-saturated ------ #
def loss_dcgan_dis(dis_fake, dis_real, ema=None, it=None):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2

def loss_dcgan_gen(dis_fake, dis_real=None):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss

# ------ lsgan ------ #
def loss_ls_dis(dis_fake, dis_real, ema=None, it=None):
  loss_real = torch.mean((dis_real + 1).pow(2))
  loss_fake = torch.mean((dis_fake - 1).pow(2))
  return loss_real, loss_fake

def loss_ls_gen(dis_fake, dis_real=None):
  return torch.mean(dis_fake.pow(2))

# ------ rahinge ------ #
def loss_rahinge_dis(dis_fake, dis_real, ema=None, it=None):
  loss_real = torch.mean(F.relu(1. - (dis_real - torch.mean(dis_fake)))/2)
  loss_fake = torch.mean(F.relu(1. + (dis_fake - torch.mean(dis_real)))/2)
  return loss_real, loss_fake

def loss_rahinge_gen(dis_fake, dis_real):
  if torch.is_tensor(dis_real):
    dis_real = torch.mean(dis_real).item()
  loss = F.relu(1 + (dis_real - torch.mean(dis_fake)))/2 + F.relu(1 - (dis_fake - dis_real))/2
  return torch.mean(loss)

# ------ hinge ------ #
def loss_hinge_dis(dis_fake, dis_real, ema=None, it=None):
  if ema is not None:
    # track the prediction
    ema.update(torch.mean(dis_fake).item(), 'D_fake', it)
    ema.update(torch.mean(dis_real).item(), 'D_real', it)

  loss_real = F.relu(1. - dis_real)
  loss_fake = F.relu(1. + dis_fake)
  return torch.mean(loss_real), torch.mean(loss_fake)

def loss_hinge_gen(dis_fake, dis_real=None):
  loss = -torch.mean(dis_fake)
  return loss

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis

