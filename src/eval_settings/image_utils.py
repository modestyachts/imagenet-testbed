import math
import random as r
import time
from functools import lru_cache
from os.path import join
import pathlib

import cv2
import numpy as np
import scipy.signal
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage.interpolation import map_coordinates
from torch.distributions import poisson
from torchvision.transforms import ToTensor


GLASS_BLUR_NUM_CACHED_SEEDS = 10
GLASS_BLUR_SEVERITIES = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2),
                         (1.5, 4, 2)]


def quantize_images(images):
    images = (images * 255).type(torch.uint8)
    images = images.type(torch.float) / 255
    return images

def gaussian_noise(image, severity):
    severity = [.08, .12, 0.18, 0.26, 0.38][severity]
    normal = torch.randn_like(image)
    image += normal * severity
    image = image.clamp(0, 1)
    return image

def shot_noise(image, severity):
    severity = [60, 25, 12, 5, 3][severity]
    image = poisson.Poisson(image * severity).sample() / severity
    image = image.clamp(0, 1)
    return image

def impulse_noise(image, severity):
    severity = [.03, .06, .09, 0.17, 0.27][severity]
    noise_mask = (torch.rand_like(image) < severity).float()
    type_mask = (torch.rand_like(image) < 0.5).float()
    image = (1 - noise_mask) * image + noise_mask * type_mask
    return image

def speckle_noise(image, severity):
    severity = [.15, .2, 0.35, 0.45, 0.6][severity]
    normal = torch.randn_like(image)
    image += image * normal * severity
    image = image.clamp(0, 1)
    return image

def contrast(image, severity):
    severity = [0.4, .3, .2, .1, .05][severity]
    means = image.mean([1, 2], keepdim=True)
    image = (image - means) * severity + means
    image = image.clamp(0, 1)
    return image

def rgb_to_hsv(image):
    out = torch.zeros_like(image)
    arr_max = image.max(0)[0]
    ipos = arr_max > 0
    delta = image.max(0)[0] - image.min(0)[0]
    s = torch.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0

    # red is max
    idx = (image[0] == arr_max) & ipos
    out[0, idx] = (image[1, idx] - image[2, idx]) / delta[idx]
    # green is max
    idx = (image[1] == arr_max) & ipos
    out[0, idx] = 2. + (image[2, idx] - image[0, idx]) / delta[idx]
    # blue is max
    idx = (image[2] == arr_max) & ipos
    out[0, idx] = 4. + (image[0, idx] - image[1, idx]) / delta[idx]

    out[0] = (out[0] / 6.0) % 1.0
    out[1] = s
    out[2] = arr_max

    return out

def hsv_to_rgb(image):
    h, s, v = image[0], image[1], image[2]
    rgb = torch.zeros_like(image)

    i = (h * 6.0).int()
    f = (h * 6.0) - i.float()
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx0 = (i % 6 == 0).float()
    idx1 = (i == 1).float()
    idx2 = (i == 2).float()
    idx3 = (i == 3).float()
    idx4 = (i == 4).float()
    idx5 = (i == 5).float()
    idxs = (s == 0).float()

    rgb[0] = v * idx0 + rgb[0] * (1 - idx0)
    rgb[1] = t * idx0 + rgb[1] * (1 - idx0)
    rgb[2] = p * idx0 + rgb[2] * (1 - idx0)

    rgb[0] = q * idx1 + rgb[0] * (1 - idx1)
    rgb[1] = v * idx1 + rgb[1] * (1 - idx1)
    rgb[2] = p * idx1 + rgb[2] * (1 - idx1)

    rgb[0] = p * idx2 + rgb[0] * (1 - idx2)
    rgb[1] = v * idx2 + rgb[1] * (1 - idx2)
    rgb[2] = t * idx2 + rgb[2] * (1 - idx2)

    rgb[0] = p * idx3 + rgb[0] * (1 - idx3)
    rgb[1] = q * idx3 + rgb[1] * (1 - idx3)
    rgb[2] = v * idx3 + rgb[2] * (1 - idx3)

    rgb[0] = t * idx4 + rgb[0] * (1 - idx4)
    rgb[1] = p * idx4 + rgb[1] * (1 - idx4)
    rgb[2] = v * idx4 + rgb[2] * (1 - idx4)

    rgb[0] = v * idx5 + rgb[0] * (1 - idx5)
    rgb[1] = p * idx5 + rgb[1] * (1 - idx5)
    rgb[2] = q * idx5 + rgb[2] * (1 - idx5)

    rgb = v * idxs + rgb * (1 - idxs)
    return rgb

def brightness(image, severity):
    severity = [.1, .2, .3, .4, .5][severity]
    image = rgb_to_hsv(image)
    image[2] += severity
    image = image.clamp(0, 1)
    image = hsv_to_rgb(image)
    return image

def saturate(image, severity):
    severity = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity]
    image = rgb_to_hsv(image)
    image[1] = image[1] * severity[0] + severity[1]
    image = image.clamp(0, 1)
    image = hsv_to_rgb(image)
    return image

@lru_cache(maxsize=6)
def disk(radius, alias_blur, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    kernel = cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
    conv_kernel = np.zeros((3, 3, *kernel.shape))
    for i in range(3):
        conv_kernel[i][i] = kernel
    conv_kernel = torch.from_numpy(conv_kernel).float()
    conv_kernel = conv_kernel.flip(2).flip(3)
    return conv_kernel

def defocus_blur(image, severity, gpu=False):
    severity = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity]
    kernel = disk(radius=severity[0], alias_blur=severity[1])
    if gpu:
        kernel = kernel.cuda()
    image = F.pad(image.unsqueeze(0), [kernel.size(-1)//2]*4, mode='reflect')
    image = F.conv2d(image, kernel)[0]
    image = image.clamp(0, 1)
    return image

@lru_cache(maxsize=20)
def gaussian_kernel(size, sigma, num_channels=3):
    x = np.linspace(- (size // 2), size // 2, size)
    x = x**2 / (2 * sigma**2)
    kernel = np.exp(- x[:, None] - x[None, :])
    kernel = kernel / kernel.sum()
    conv_kernel = np.zeros((num_channels, num_channels, *kernel.shape))
    for i in range(num_channels):
        conv_kernel[i][i] = kernel
    return torch.from_numpy(conv_kernel).float()



def gaussian_blur_helper(image, size, sigma):
    kernel = gaussian_kernel(size, sigma, num_channels=image.shape[0]).to(
        image.device).type(image.dtype)
    image = F.pad(image.unsqueeze(0), [kernel.size(-1) // 2] * 4,
                  mode='reflect')
    return F.conv2d(image, kernel)[0]

def gaussian_blur_separated(image, size, sigma):
    """
    >>> image = torch.rand(3, 5, 5)
    >>> expected = gaussian_blur_helper(image, 3, 1)
    >>> real = gaussian_blur_separated(image, 3, 1)
    >>> assert torch.allclose(expected, real), (
    ...     f"Expected:\\n{expected}\\nSaw:\\n{real}")
    """
    kernel_1d = scipy.signal.gaussian(size, sigma)
    kernel_1d /= kernel_1d.sum()
    c = image.shape[0]
    conv1d_x = image.new_zeros(c, c, size, 1)
    for c_i in range(c):
        conv1d_x[c_i, c_i, :, 0] = torch.from_numpy(kernel_1d)
    image = F.pad(image.unsqueeze(0), [size // 2] * 4, mode='reflect')
    image = F.conv2d(image, conv1d_x)
    image = F.conv2d(image, conv1d_x.permute((0, 1, 3, 2)))
    return image[0]

def gaussian_blur(image, severity):
    severity = [1, 2, 3, 4, 6][severity]
    image = gaussian_blur_helper(image, severity * 4 - 1, severity)
    image = image.clamp(0, 1)
    return image


def _glass_blur_indices_from_deltas(image_shape, max_delta, num_iters, deltas):
    """
    >>> image = np.random.rand(224, 224, 3).astype('float32')
    >>> severity = (0.7, 1, 2)
    >>> s, n = severity[1], severity[2]
    >>> deltas = np.random.randint(-s, s, size=(3, 224 - s, 224 - s, 2))

    >>> indices = _glass_blur_indices_from_deltas(image.shape[1], s, n, deltas)
    >>> direct_image = image[indices[:, :, 0], indices[:, :, 1]]
    >>> manual_image = image.copy()
    >>> _glass_blur_manual_index_with_deltas_(manual_image, deltas, severity)
    >>> assert np.allclose(manual_image, direct_image)
    """
    heights = list(range(image_shape - max_delta, max_delta, -1))
    widths = list(range(image_shape - max_delta, max_delta, -1))
    # Create x and y indexing tensor for image of size (height, width)
    # xs: [[0, 1, 2, ..., w], [0, 1, ..., w], ..., [0, 1, ..., w]]
    # ys: [[0, 1, 2, ..., h], [0, 1, ..., h], ..., [0, 1, ..., h]]^T
    xs = torch.stack((torch.arange(image_shape),) * image_shape)
    ys = torch.stack((torch.arange(image_shape),) * image_shape).t()
    indices = torch.stack((ys, xs), dim=2).numpy() # (h, w, 2)
    for i in range(num_iters):
        for h_i, h in enumerate(heights):
            for w_i, w in enumerate(widths):
                dx, dy = deltas[i, h_i, w_i]
                h_prime, w_prime = h + dy, w + dx
                indices[h, w], indices[h_prime, w_prime] = (
                    indices[h_prime, w_prime], indices[h, w])
    return indices


# Create a cache that caches an index for each seed and severity.
@lru_cache(maxsize=min(
    GLASS_BLUR_NUM_CACHED_SEEDS * len(GLASS_BLUR_SEVERITIES), 256))
def _glass_blur_compute_indices(image_shape, max_delta, num_iters, seed=None):
    heights = list(range(image_shape - max_delta, max_delta, -1))
    widths = list(range(image_shape - max_delta, max_delta, -1))
    rs = np.random.RandomState(seed)
    deltas = rs.randint(-max_delta,
                        max_delta,
                        size=(num_iters, len(heights), len(widths), 2))
    return _glass_blur_indices_from_deltas(image_shape, max_delta, num_iters,
                                           deltas)

def _glass_blur_direct_index(image, severity, seed=None):
    indices = _glass_blur_compute_indices(image.shape[1],
                                          severity[1],
                                          severity[2],
                                          seed=seed)
    return image[indices[:, :, 0], indices[:, :, 1]]

def glass_blur(image, severity):
    severity = GLASS_BLUR_SEVERITIES[severity]
    kernel = gaussian_kernel(5, severity[0]*2)
    image = F.pad(image.unsqueeze(0), [kernel.size(-1)//2]*4, mode='reflect')
    image = F.conv2d(image, kernel)[0]
    image = image.clamp(0, 1)

    image = image.cpu().numpy().transpose(1, 2, 0)
    seed = np.random.randint(GLASS_BLUR_NUM_CACHED_SEEDS)
    image = _glass_blur_direct_index(image, severity, seed=seed)

    image = torch.from_numpy(image.transpose(2, 0, 1))
    image = F.pad(image.unsqueeze(0), [kernel.size(-1)//2]*4, mode='reflect')
    image = F.conv2d(image, kernel)[0]
    image = image.clamp(0, 1)
    return image

def spatter(image, severity):
    severity = [(0.65, 0.3, 4, 0.69, 0.6, 0),
               (0.65, 0.3, 3, 0.68, 0.6, 0),
               (0.65, 0.3, 2, 0.68, 0.5, 0),
               (0.65, 0.3, 1, 0.65, 1.5, 1),
               (0.67, 0.4, 1, 0.65, 1.5, 1)][severity]

    liquid_layer = torch.randn(1, 1, *image.size()[1:])
    liquid_layer = liquid_layer * severity[1] + severity[0]
    kernel = gaussian_kernel(severity[2]*4-1, severity[2], num_channels=1)
    liquid_layer = F.pad(liquid_layer, [kernel.size(-1)//2]*4, mode='reflect')
    liquid_layer = F.conv2d(liquid_layer, kernel)[0][0]
    liquid_layer[liquid_layer < severity[3]] = 0

    if severity[5] == 0:
        liquid_l = (liquid_layer * 255).byte().cpu().numpy()
        dist = 255 - cv2.Canny(liquid_l, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        dist[dist > 20] = 20
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        dist = torch.from_numpy(dist).float()

        kernel = torch.tensor([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]).float()
        dist = F.pad(dist.view(1, 1, *dist.size()), [1, 1, 1, 1], mode='reflect')
        dist = F.conv2d(dist, kernel.view(1, 1, *kernel.size()))
        dist[dist < 0] = 0
        kernel = torch.ones(1, 1, 3, 3).float() / 9
        dist = F.pad(dist, [1, 1, 1, 1], mode='reflect')
        dist = F.conv2d(dist, kernel)[0][0]

        m = liquid_layer * dist
        m /= m.max()
        m = m.repeat(3, 1, 1)
        m *= severity[4]

        color = torch.tensor([175/255, 238/255, 238/255])
        color = color.view(3, 1, 1).repeat(1, *image.size()[1:])
        image = image + m * color
        image = image.clamp(0, 1)

    else:
        m = (liquid_layer > severity[3]).float().unsqueeze(0).unsqueeze(0)
        kernel = gaussian_kernel(math.ceil(severity[4]*4-1), severity[4], num_channels=1)
        m = F.pad(m, [kernel.size(-1)//2]*4, mode='reflect')
        m = F.conv2d(m, kernel)[0][0]
        m[m < 0.8] = 0

        color = torch.tensor([63/255, 42/255, 20/255])
        color = color.view(3, 1, 1).repeat(1, *image.size()[1:])
        image = image * (1 - m) + color * m
        image = image.clamp(0, 1)

    return image

def plasma_fractal(mapsize=256, wibbledecay=3):
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    maparray = maparray / maparray.max()
    return torch.from_numpy(maparray).float()

def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def fog(image, severity):
    severity = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity]
    max_val = image.max()
    img_size = max(image.size())
    mapsize = next_power_of_2(img_size)
    fog = plasma_fractal(mapsize=mapsize, wibbledecay=severity[1])[:img_size, :img_size]
    image += severity[0] * fog
    image *= max_val / (max_val + severity[0])
    image = image.clamp(0, 1)
    return image

def jpeg_compression(image, severity):
    severity = [25, 18, 15, 10, 7][severity]
    param = [int(cv2.IMWRITE_JPEG_QUALITY), severity]
    image = (image * 255).type(torch.uint8).permute(1, 2, 0)
    _, encimg = cv2.imencode('.jpg', image.cpu().numpy(), param)
    image = torch.from_numpy(cv2.imdecode(encimg, 1))
    image = (image.float() / 255).permute(2, 0, 1)
    return image

def pixelate(image, severity):
    severity = [0.6, 0.5, 0.4, 0.3, 0.25][severity]
    _, h, w = image.size()
    image = F.interpolate(image.unsqueeze(0), size=(int(h*severity), int(w*severity)), mode='area')
    image = F.interpolate(image, size=(h, w), mode='nearest')[0]
    return image

@lru_cache(maxsize=100)
def get_frost_file(i, img_size):
    file = str((pathlib.Path(__file__).parent / 'imagenet-c_frost_pictures').resolve())
    file = join(file, f"frost{i}.{'png' if i <=3 else 'jpg'}")
    frost = cv2.imread(file)
    scale_factor = max(1, img_size / min(frost.shape[0], frost.shape[1]))
    size = (int(np.ceil(frost.shape[0] * scale_factor)), int(np.ceil(frost.shape[1] * scale_factor)))
    frost = cv2.resize(frost, dsize=size, interpolation=cv2.INTER_CUBIC)
    file = ToTensor()(frost[..., [2, 1, 0]])
    return file

def frost(image, severity):
    severity = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity]
    start = time.time()
    img_size = image.size(1)
    frost = get_frost_file(r.randint(1, 6), img_size)
    x_start, y_start = r.randint(0, frost.size(1) - img_size), r.randint(0, frost.size(2) - img_size)
    frost = frost[:, x_start:x_start + img_size, y_start:y_start + img_size]
    image = severity[0] * image + severity[1] * frost
    image = image.clamp(0, 1)
    return image

def motion_blur_generate_kernel(radius, angle, sigma):
    """
    Args:
        radius
        angle (float): Radians clockwise from the (x=1, y=0) vector. This
            is how ImageMagick's -motion-blur filter accepts angles, as far
            as I can tell.

    >>> mb_1_0_inf_expected = torch.ones(3) / 3
    >>> mb_1_0_inf = motion_blur_generate_kernel(1, 0, np.inf)[0]
    >>> assert torch.all(torch.isclose(mb_1_0_inf[0], mb_1_0_inf_expected))

    >>> g_3_1 = torch.from_numpy(scipy.signal.gaussian(5, 1)[2:]).float()
    >>> g_3_1 /= g_3_1.sum()

    >>> mb_1_0_1 = motion_blur_generate_kernel(1, 0, 1)[0]
    >>> assert torch.all(mb_1_0_1[0] == g_3_1), (mb_1_0_1[0], g_3_1)
    >>> assert torch.all(mb_1_0_1[1] == 0)
    >>> assert torch.all(mb_1_0_1[2] == 0)
    """
    # Make angles be counterclockwise from (x=1, y=0) vector to maintain sanity.
    angle = 2 * np.pi - angle

    # Make all angles lie in [0, 2*pi]
    if angle < 0:
        angle += math.ceil(angle / (2 * np.pi)) * 2*np.pi
    if angle > 2 * np.pi:
        angle = angle % (2 * np.pi)

    size = 2 * radius + 1
    kernel = torch.zeros((size, size))
    # Gaussian centered at 0th element.
    kernel_1d = scipy.signal.gaussian(size * 2 - 1, sigma)[size-1:]

    direction_up = 0 <= angle <= np.pi
    direction_right = (angle < np.pi / 2) or (angle > 3 / 2 * np.pi)

    cy = size - 1 if direction_up else 0
    cx = 0 if direction_right else size - 1

    # dy is relative to matrix coordinates, so, e.g., angle of np.pi/4 should
    # be a line going up => dy should be negative.
    dx, dy = np.cos(angle).item(), -np.sin(angle).item()
    for i in range(size):
        # *o*ffset_*x*, *o*ffset_*y*
        ox, oy = dx * i, dy * i
        x = min(cx + round(ox), size)
        y = min(cy + round(oy), size)
        assert x >= 0, f'x={x} should be >= 0!'
        assert y >= 0, f'y={y} should be >= 0!'
        kernel[y, x] = kernel_1d[i]
    kernel /= kernel.sum()
    return kernel, cy, cx

def motion_blur(image, severity, gpu=False):
    radius, sigma = [(10, 3), (15, 5), (15, 8), (15, 12),
                     (20, 15)][severity]
    angle = np.random.uniform(-np.pi / 4, np.pi / 4)

    _, image_h, image_w = image.shape
    # https://github.com/ImageMagick/ImageMagick/blob/829452165a92db61b5e3fdb7f8a3e8f728f7e8ac/MagickCore/effect.c#L2051
    kernel, cy, cx = motion_blur_generate_kernel(radius, angle, sigma)

    # Pad so as to re-center image after blur.
    size = kernel.shape[0]
    pad_x = (0, size-1) if cx == 0 else (size-1, 0)
    pad_y = (0, size-1) if cy == 0 else (size-1, 0)

    # # Convert to 3-channel filter
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1]).repeat((3, 1, 1, 1))
    if gpu:
        kernel = kernel.cuda()
    image = F.pad(image.unsqueeze(0), pad_x + pad_y, mode='replicate')
    output = F.conv2d(image, kernel, groups=3).squeeze(0)
    return output

def clipped_zoom(image, zoom_factor):
    h = image.size(1)
    ch = int(np.ceil(h / float(zoom_factor)))
    top = (h - ch) // 2
    image = image[:, top:top + ch, top:top + ch]
    image = F.interpolate(image.unsqueeze(0), scale_factor=zoom_factor, mode='bilinear')[0]
    trim_top = (image.size(1) - h) // 2
    image = image[:, trim_top:trim_top + h, trim_top:trim_top + h]
    return image

def zoom_blur(image, severity):
    severity = [np.arange(1, 1.11, 0.01),
                np.arange(1, 1.16, 0.01),
                np.arange(1, 1.21, 0.02),
                np.arange(1, 1.26, 0.02),
                np.arange(1, 1.31, 0.03)][severity]

    out = torch.zeros_like(image)
    for zoom_factor in severity:
        out += clipped_zoom(image, zoom_factor)

    image = (image + out) / (len(severity) + 1)
    image = image.clamp(0, 1)
    return image

def greyscale(image):
    weights = image.new([0.299, 0.587, 0.114]).view(3, 1, 1)
    image = (image * weights).sum(dim=0, keepdim=True)
    image = image.repeat(3, 1, 1)
    return image

def snow(image, severity, gpu=False):
    severity = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
                (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
                (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
                (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
                (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity]

    snow_layer = torch.randn(1, image.size(1), image.size(2)) * severity[1] + severity[0]
    snow_layer = clipped_zoom(snow_layer, severity[2])
    snow_layer[snow_layer < severity[3]] = 0

    kernel, cy, cx = motion_blur_generate_kernel(severity[4], np.random.uniform(-135, -45), severity[5])
    size = kernel.shape[0]
    pad_x = (0, size-1) if cx == 0 else (size-1, 0)
    pad_y = (0, size-1) if cy == 0 else (size-1, 0)
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
    if gpu:
        kernel = kernel.cuda()
        snow_layer = snow_layer.cuda()
    snow_layer = F.pad(snow_layer.unsqueeze(0), pad_x + pad_y, mode='replicate')
    snow_layer = F.conv2d(snow_layer, kernel).squeeze(0)

    image = severity[6] * image + (1 - severity[6]) * torch.max(image, greyscale(image) * 1.5 + 0.5)
    image = image + snow_layer + snow_layer.flip(2).flip(1)
    image = image.clamp(0, 1)
    return image


def elastic_transform(image, severity, gpu=False):
    image = image.permute((1, 2, 0))

    image = image.cpu().numpy()
    shape = image.shape
    h, w = shape[:2]

    c = [
        # 244 should have been 224, but ultimately nothing is incorrect
        (244 * 2, 244 * 0.7, 244 * 0.1),
        (244 * 2, 244 * 0.08, 244 * 0.2),
        (244 * 0.05, 244 * 0.01, 244 * 0.02),
        (244 * 0.07, 244 * 0.01, 244 * 0.02),
        (244 * 0.12, 244 * 0.01, 244 * 0.02)
    ][severity]

    # random affine
    center_square = np.float32((h, w)) // 2
    square_size = min((h, w)) // 3
    pts1 = np.float32([
        center_square + square_size,
        [center_square[0] + square_size, center_square[1] - square_size],
        center_square - square_size
    ])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(
        np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    image_th = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(0)
    if gpu:
        image_th = image_th.cuda()

    # Generate a kernel matching scipy's gaussian filter
    # https://github.com/scipy/scipy/blob/e1e44d12637997606b1bcc0c6de232349e11eee0/scipy/ndimage/filters.py#L214
    sigma = c[1]
    truncate = 3
    radius = min(int(truncate * sigma + 0.5), h)

    deltas = torch.FloatTensor(2, h, w).uniform_(-1, 1)
    if gpu:
        deltas = deltas.cuda()
    deltas = gaussian_blur_separated(deltas, 2 * radius - 1, sigma) * c[0]
    dx, dy = deltas[0], deltas[1]

    dx = dx.squeeze(0).unsqueeze(-1).float()
    dy = dy.squeeze(0).unsqueeze(-1).float()

    # y : [[0, 0, 0, 0], [1, 1, 1, 1], ...]
    # x : [[0, 1, 2, 3], [0, 1, 2, 3], ...]
    y, x = torch.meshgrid(torch.arange(w), torch.arange(h))
    x = x.unsqueeze(-1).to(dx.device).float()
    y = y.unsqueeze(-1).to(dy.device).float()
    indices = torch.stack((x + dx, y + dy), dim=-1)
    indices = indices.permute((2, 0, 1, 3))
    indices[..., 0] = ((indices[..., 0] / h) - 0.5) * 2
    indices[..., 1] = ((indices[..., 1] / w) - 0.5) * 2
    output = F.grid_sample(image_th,
                           indices,
                           mode='bilinear',
                           padding_mode='reflection').clamp(0, 1).squeeze(0)
    
    return output


corruption_tuple = (brightness, contrast, defocus_blur, elastic_transform, fog, frost, gaussian_blur,
                    gaussian_noise, impulse_noise, jpeg_compression, motion_blur, pixelate, saturate,
                    shot_noise, snow, spatter, speckle_noise, zoom_blur, greyscale, quantize_images)
corruption_dict = {corr_func.__name__: corr_func for corr_func in corruption_tuple}
