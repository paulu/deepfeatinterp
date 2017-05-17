
import numpy as np

# Credit: jcjohnson/cnn-vis
def tv_norm(x, beta=2.0, verbose=False, operator='naive'):
  """
  Compute the total variation norm and its gradient.
  
  The total variation norm is the sum of the image gradient
  raised to the power of beta, summed over the image.
  We approximate the image gradient using finite differences.
  We use the total variation norm as a regularizer to encourage
  smoother images.
  Inputs:
  - x: numpy array of shape (1, C, H, W)
  Returns a tuple of:
  - loss: Scalar giving the value of the norm
  - dx: numpy array of shape (1, C, H, W) giving gradient of the loss
        with respect to the input x.
  """
  assert x.shape[0] == 1
  if operator == 'naive':
    x_diff = x[:, :, :-1, :-1] - x[:, :, :-1, 1:]
    y_diff = x[:, :, :-1, :-1] - x[:, :, 1:, :-1]
  elif operator == 'sobel':
    x_diff  =  x[:, :, :-2, 2:]  + 2 * x[:, :, 1:-1, 2:]  + x[:, :, 2:, 2:]
    x_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, 1:-1, :-2] + x[:, :, 2:, :-2]
    y_diff  =  x[:, :, 2:, :-2]  + 2 * x[:, :, 2:, 1:-1]  + x[:, :, 2:, 2:]
    y_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, :-2, 1:-1] + x[:, :, :-2, 2:]
  elif operator == 'sobel_squish':
    x_diff  =  x[:, :, :-2, 1:-1]  + 2 * x[:, :, 1:-1, 1:-1]  + x[:, :, 2:, 1:-1]
    x_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, 1:-1, :-2] + x[:, :, 2:, :-2]
    y_diff  =  x[:, :, 1:-1, :-2]  + 2 * x[:, :, 1:-1, 1:-1]  + x[:, :, 1:-1, 2:]
    y_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, :-2, 1:-1] + x[:, :, :-2, 2:]
  else:
    assert False, 'Unrecognized operator %s' % operator
  grad_norm2 = x_diff ** 2.0 + y_diff ** 2.0
  grad_norm2[grad_norm2 < 1e-3] = 1e-3
  grad_norm_beta = grad_norm2 ** (beta / 2.0)
  loss = np.sum(grad_norm_beta)
  dgrad_norm2 = (beta / 2.0) * grad_norm2 ** (beta / 2.0 - 1.0)
  dx_diff = 2.0 * x_diff * dgrad_norm2
  dy_diff = 2.0 * y_diff * dgrad_norm2
  dx = np.zeros_like(x)
  if operator == 'naive':
    dx[:, :, :-1, :-1] += dx_diff + dy_diff
    dx[:, :, :-1, 1:] -= dx_diff
    dx[:, :, 1:, :-1] -= dy_diff
  elif operator == 'sobel':
    dx[:, :, :-2, :-2] += -dx_diff - dy_diff
    dx[:, :, :-2, 1:-1] += -2 * dy_diff
    dx[:, :, :-2, 2:] += dx_diff - dy_diff
    dx[:, :, 1:-1, :-2] += -2 * dx_diff
    dx[:, :, 1:-1, 2:] += 2 * dx_diff
    dx[:, :, 2:, :-2] += dy_diff - dx_diff
    dx[:, :, 2:, 1:-1] += 2 * dy_diff
    dx[:, :, 2:, 2:] += dx_diff + dy_diff
  elif operator == 'sobel_squish':
    dx[:, :, :-2, :-2] += -dx_diff - dy_diff
    dx[:, :, :-2, 1:-1] += dx_diff -2 * dy_diff
    dx[:, :, :-2, 2:] += -dy_diff
    dx[:, :, 1:-1, :-2] += -2 * dx_diff + dy_diff
    dx[:, :, 1:-1, 1:-1] += 2 * dx_diff + 2 * dy_diff
    dx[:, :, 1:-1, 2:] += dy_diff
    dx[:, :, 2:, :-2] += -dx_diff
    dx[:, :, 2:, 1:-1] += dx_diff

  def helper(name, x):
    num_nan = np.isnan(x).sum()
    num_inf = np.isinf(x).sum()
    num_zero = (x == 0).sum()
    print '%s: NaNs: %d infs: %d zeros: %d' % (name, num_nan, num_inf, num_zero)
  
  if verbose:
    print '-' * 40
    print 'tv_norm debug output'
    helper('x', x)
    helper('x_diff', x_diff)
    helper('y_diff', y_diff)
    helper('grad_norm2', grad_norm2)
    helper('grad_norm_beta', grad_norm_beta)
    helper('dgrad_norm2', dgrad_norm2)
    helper('dx_diff', dx_diff)
    helper('dy_diff', dy_diff)
    helper('dx', dx)
    print
  
  return loss, dx

