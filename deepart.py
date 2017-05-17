import os

import sys
import numpy as np
import skimage
from scipy import optimize
import os
import os.path

import totalvariation

def ensuredir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def save_image_blob(filepath, net, data):
    deproc_img = net.transformer.deprocess(net.inputs[0], data)
    deproc_img = np.clip(deproc_img, 0, 1)
    skimage.io.imsave(filepath, deproc_img)


def comp_fet_mx(blob_data):
    fet_mx = np.reshape(
        blob_data[0],
        (blob_data.shape[1], blob_data.shape[2] * blob_data.shape[3])
    )
    return fet_mx


def comp_gram(blob_data):
    fet_mx = comp_fet_mx(blob_data)
    return np.dot(fet_mx, fet_mx.T)


def style_grad(gen_data, target_data):
    # Note: target_data should already be the gram matrix!
    gram_mx_A = target_data
    gram_mx_G = comp_gram(gen_data)
    local_add = gram_mx_G - gram_mx_A

    fet_mx = comp_fet_mx(gen_data)
    grad = np.dot(fet_mx.T, local_add).T
    grad = np.reshape(grad, gen_data.shape)
    loss = np.sum(local_add ** 2) / 4

    norm = gen_data.size
    loss /= norm
    grad /= norm

    return loss, grad


def content_grad(gen_data, target_data, weight=None):
    if weight is None:
        grad = gen_data - target_data
        loss = np.sum(grad ** 2) * 0.5
    else:
        grad = gen_data - target_data
        loss = np.sum((grad ** 2) * weight) * 0.5
        grad *= weight

    return loss, grad


def gradient_grad(gen_data, target_data, weight):
    # gen_data is M x K x H x W image
    # target_data is 2 x M x K x H x W target fy, fx (last row and column are ignored)
    # weight is M x 1 x H x W alpha mask

    fy = np.diff(gen_data, axis=2)
    fx = np.diff(gen_data, axis=3)
    fymgy = fy - target_data[0,:,:,:-1,:]
    fxmgx = fx - target_data[1,:,:,:,:-1]
    # fx = 1D finite diff
    # partial derivative is -1 => take reversed finite diff of the finite diff,
    # and append -first and +last element of fx
    grady = np.concatenate([-fy[:,:,:1,:], np.diff(fymgy[:,:,::-1,:], axis=2)[:,:,::-1,:], fy[:,:,-1:,:]], axis=2)
    gradx = np.concatenate([-fx[:,:,:,:1], np.diff(fxmgx[:,:,:,::-1], axis=3)[:,:,:,::-1], fx[:,:,:,-1:]], axis=3)
    lossy = np.sum((fymgy ** 2) * weight[:,:,:-1,:]) * 0.5
    lossx = np.sum((fxmgx ** 2) * weight[:,:,:,:-1]) * 0.5

    return lossy + lossx, (grady + gradx) * weight


def gen_target_data(root_dir, caffe, net, targets):
    if not root_dir is None:
        ensuredir(root_dir)

    target_data_list = []
    for target_i, (target_img_path, target_blob_names, is_gram, _) in enumerate(targets):
        # Load and rescale to [0, 1]
        if isinstance(target_img_path,str):
            target_img = caffe.io.load_image(target_img_path)
        else:
            target_img = target_img_path
            assert target_img.dtype == np.float32
            assert target_img.ndim == 3
            assert target_img.min() >= 0
            assert target_img.max() <= 1
        caffe_in = net.preprocess_inputs([target_img], auto_reshape=True)
        # Copy image into input blob
        get_data_blob(net).data[...] = caffe_in
        net.forward()
        target_datas = {}
        for target_blob_name in target_blob_names:
            target_data = net.blobs[target_blob_name].data.copy()
            # Apply ReLU
            pos_mask = target_data > 0
            target_data[~pos_mask] = 0
            if is_gram:
                target_datas[target_blob_name] = comp_gram(target_data)
            else:
                target_datas[target_blob_name] = target_data

        target_data_list.append(target_datas)

        if not root_dir is None:
            save_image_blob(
                os.path.join(root_dir, 'target-{}.jpg'.format(target_i)),
                net,
                get_data_blob(net).data[0],
            )

    return target_data_list

def objective_func(x, net, all_target_blob_names, targets, target_data_list, tv_lambda, tv_beta):
    # Makes one iteration step and updates the gradient of the data blob

    get_data_blob(net).data[...] = np.reshape(x, get_data_blob(net).data.shape)
    get_data_blob(net).diff[...] = 0
    net.forward()

    loss = 0
    # Go through target blobs in reversed order
    for i in range(len(all_target_blob_names)):
        blob_i = len(all_target_blob_names) - 1 - i
        start = all_target_blob_names[blob_i]

        if blob_i == 0:
            end = None
        else:
            end = all_target_blob_names[blob_i - 1]

        # Get target blob
        target_blob = net.blobs[start]
        if i == 0:
            target_blob.diff[...] = 0

        gen_data = target_blob.data.copy()
        # Apply RELU
        pos_mask = gen_data > 0
        gen_data[~pos_mask] = 0

        # Go through all images and compute accumulated gradient for the current target blob
        target_blob_add_diff = np.zeros_like(target_blob.diff, dtype=np.float64)
        for target_i, (_, target_blob_names, is_gram, weight) in enumerate(targets):
            # Skip if the current blob is not among the target's blobs
            if start not in target_blob_names:
                continue

            target_data = target_data_list[target_i][start]
            if is_gram:
                c_loss, c_grad = style_grad(gen_data, target_data)
            else:
                c_loss, c_grad = content_grad(gen_data, target_data)

            # Apply RELU
            c_grad[~pos_mask] = 0
            target_blob_add_diff += c_grad * weight / len(target_blob_names)
            loss += c_loss * weight / len(target_blob_names)

        target_blob.diff[...] += target_blob_add_diff
        net.backward(start=start, end=end)

    if tv_lambda > 0:
        tv_loss, tv_grad = totalvariation.tv_norm(x.reshape(get_data_blob(net).data.shape),beta=tv_beta)
        return loss + tv_loss*tv_lambda, np.ravel(get_data_blob(net).diff).astype(np.float64) + np.ravel(tv_grad)*tv_lambda
    else:
        return loss, np.ravel(get_data_blob(net).diff).astype(np.float64)

def objective_func2(x, net, blob_names, target, target_data, tv_lambda, tv_beta):
    '''
    blob_names is a list of all blobs sorted front-to-back
    target and target_data are dicts of blob names
    target[blob name] is a list of 3-tuples (weight, target type, data index)
      weight is a scalar
      target type is 'gram' or 'l2'
      data index is a zero-based index into the target tensor
    target_data[blob name] is a tensor (image x channel x height x width)
    tv_lambda and tv_beta are scalars for the total variation regularizer
    '''

    # Makes one iteration step and updates the gradient of the data blob

    get_data_blob(net).data[...] = np.reshape(x, get_data_blob(net).data.shape)
    get_data_blob(net).diff[...] = 0
    net.forward()

    loss = 0
    # Go through target blobs in reversed order
    for i in range(len(blob_names)):
        blob_i = len(blob_names) - 1 - i
        start = blob_names[blob_i]

        if blob_i == 0:
            end = None
        else:
            end = blob_names[blob_i - 1]

        # Get target blob
        target_blob = net.blobs[start]
        if i == 0:
            target_blob.diff[...] = 0
        gen_data = target_blob.data.copy()

        # Apply RELU
        pos_mask = gen_data > 0
        gen_data[~pos_mask] = 0

        # Accumulate objectives
        target_blob_add_diff = np.zeros_like(target_blob.diff, dtype=np.float64)
        for (weight, target_type, data_i) in target[start]:
            if target_type=='gram':
                c_loss, c_grad = style_grad(gen_data, target_data[start][data_i])
            elif target_type=='l2':
                c_loss, c_grad = content_grad(gen_data, target_data[start][data_i])
            else:
                raise ValueError('Unknown target type: {}'.format(target_type))

            # Apply RELU
            c_grad[~pos_mask] = 0
            target_blob_add_diff += c_grad * weight
            loss += c_loss * weight

        # Update model
        target_blob.diff[...] += target_blob_add_diff
        net.backward(start=start, end=end)

    if tv_lambda > 0:
        tv_loss, tv_grad = totalvariation.tv_norm(x.reshape(get_data_blob(net).data.shape), beta=tv_beta)
        return loss + tv_loss * tv_lambda, np.ravel(get_data_blob(net).diff).astype(np.float64) + np.ravel(tv_grad) * tv_lambda
    else:
        return loss, np.ravel(get_data_blob(net).diff).astype(np.float64)


def get_data_blob(net):
    return net.blobs[net.inputs[0]]


def set_data(net, init_img):
    caffe_in = net.preprocess_inputs([init_img], auto_reshape=True)
    # Copy image into input blob
    get_data_blob(net).data[...] = caffe_in


class DisplayFunctor():
    def __init__(self, net, root_dir, display):
        self.net = net
        self.root_dir = root_dir
        self.display = display
        self.it = 0

    def __call__(self, x):
        if self.it % self.display == 0:
            #print 'Saving image for iteration {}...'.format(self.it)
            save_image_blob(
                os.path.join(self.root_dir, '{:06}-it.jpg'.format(self.it)),
                self.net,
                np.reshape(x, get_data_blob(self.net).data.shape)[0],
            )

        self.it += 1


def optimize_img(init_img, solver_type, solver_param, max_iter, display, root_dir, net,
                 all_target_blob_names, targets, target_data_list, tv_lambda=0.001):
    ensuredir(root_dir)

    solver_param.update({
        'maxiter': max_iter,
        'disp': True,
    })

    # Set initial value and reshape net
    set_data(net, init_img)
    x0 = np.ravel(init_img).astype(np.float64)

    mins = np.full_like(x0, -128)
    maxs = np.full_like(x0, 162)

    bounds = zip(mins, maxs)
    display_func = DisplayFunctor(net, root_dir, display)

    tv_beta = 2
    opt_res = optimize.minimize(
        objective_func,
        x0,
        args=(net, all_target_blob_names, targets, target_data_list, tv_lambda, tv_beta),
        bounds=bounds,
        method=solver_type,
        jac=True,
        callback=display_func,
        options=solver_param,
    )
    #print opt_res

    data = np.reshape(opt_res.x, get_data_blob(net).data.shape)[0]
    deproc_img = net.transformer.deprocess(net.inputs[0], data)
    deproc_img = np.clip(deproc_img, 0, 1)
    return deproc_img
