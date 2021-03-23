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

﻿"""LeCam Loss function."""

import tensorflow as tf
from tensorflow.python.training import moving_averages


def lecamreg(D_real, D_fake):
  """Compute the lecam regularization loss.

  Args:
    D_real: discriminator predictions of real images.
    D_fake: discriminator predictions of fake images.
  Returns:
    D_lecam_loss: lecam regularization loss.
  """

  # Historical statics of the descriminator predictions
  real_scores_mean = tf.stop_gradient(tf.reduce_mean(D_real.scores))
  fake_scores_mean = tf.stop_gradient(tf.reduce_mean(D_fake.scores))
  ema_real_scores_var = tf.Variable(
      0.0,
      name='ema_real_scores',
      shape=(),
      dtype=tf.float32,
      synchronization=tf.VariableSynchronization.ON_READ,
      trainable=False,
      aggregation=tf.VariableAggregation.MEAN)
  ema_fake_scores_var = tf.Variable(
      0.0,
      name='ema_fake_scores',
      shape=(),
      dtype=tf.float32,
      synchronization=tf.VariableSynchronization.ON_READ,
      trainable=False,
      aggregation=tf.VariableAggregation.MEAN)
  ema_real_scores_cur = moving_averages.assign_moving_average(
      ema_real_scores_var, real_scores_mean, decay=0.99, zero_debias=True)
  ema_fake_scores_cur = moving_averages.assign_moving_average(
      ema_fake_scores_var, fake_scores_mean, decay=0.99, zero_debias=True)

  # LeCam regularization loss
  with tf.name_scope('Loss_lecam'):
    D_lecam_real = tf.reduce_mean(tf.square(tf.nn.relu(D_real.scores - ema_fake_scores_cur)))
    D_lecam_fake = tf.reduce_mean(tf.square(tf.nn.relu(ema_real_scores_cur - D_fake.scores)))
    D_lecam_loss = D_lecam_real + D_lecam_fake
  return D_lecam_loss
