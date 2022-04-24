import numpy as np
import torch
import torch.nn.functional as F
from utils.gaussian_blur import gaussian_blur
import pbsa
from geomloss import SamplesLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def normalize(t, mean, std):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


def input_diversity(img):
    rnd = torch.randint(224, 257, (1,)).item()
    rescaled = F.interpolate(img, (rnd, rnd), mode='nearest')
    h_rem = 256 - rnd
    w_hem = 256 - rnd
    pad_top = torch.randint(0, h_rem + 1, (1,)).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_hem + 1, (1,)).item()
    pad_right = w_hem - pad_left
    padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
    padded = F.interpolate(padded, (224, 224), mode='nearest')
    return padded


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, padding_size):
    # x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = F.conv2d(x, stack_kern, padding=(padding_size, padding_size), groups=3)
    return x


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


stack_kern, padding_size = project_kern(3)


def local_adv(model, criterion, img, label, eps, attack_type, iters, mean, std, index, m, k, p, apply_ti=False, skip=True, amp=10):
    adv = img.detach()
    if attack_type == 'rfgsm':
        alpha = 2 / 255
        adv = adv + alpha * torch.randn(img.shape).cuda().detach().sign()
        eps = eps - alpha

    adv.requires_grad = True

    #
    if index == 'pbsa':
        good_layer, weight = pbsa.pma(img, model, criterion, label, mean, std, m, p, k)
        if skip:
            h = pbsa.register_hook_for_all(model, gamma=0.2)
    if index != 'pbsa' and skip:
        h = pbsa.register_hook_for_all(model, gamma=0.2)


    if attack_type in ['fgsm', 'rfgsm']:
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations

    if attack_type == 'pifgsm':
        # alpha = step = eps / iterations
        alpha_beta = step * amp
        gamma = alpha_beta
        amplification = 0.0

    adv_noise = 0
    # cos_criterion = torch.nn.CosineEmbeddingLoss(margin=-1.0).cuda()
    for j in range(iterations):
        if attack_type == 'dim':
            adv_r = input_diversity(adv)
        else:
            adv_r = adv

        out_adv, att_adv = model(normalize(adv_r.clone(), mean=mean, std=std))
        out_img, att_img = model(normalize(img.clone(), mean=mean, std=std))
        loss = 0

        if isinstance(out_adv, list) and index == 'pma':  # Jean
            loss = 0
            for idx in range(k):
                layer = torch.squeeze(good_layer)[idx].item()
                loss += criterion(out_adv[layer], label)
                # loss += criterion(out_adv[layer], label) + 0.01*torch.norm(att_adv[layer]-att_img[layer])
            for j in range(12):
                if j not in good_layer:
                    loss += 0.01 * torch.norm(att_adv[j] - att_img[j])
            loss = loss + 0.5 * torch.norm((adv_r - img))

        elif isinstance(out_adv, list) and index == 'all':
            loss = 0
            for idx in range(len(out_adv)):
                loss += criterion(out_adv[idx], label)
        elif isinstance(out_adv, list) and index == 'last':
            loss = criterion(out_adv[-1], label)
        else:
            for j in range(12):
                loss+=0.01 * torch.norm(att_adv[j] - att_img[j])
            loss = loss + 0.5 * torch.norm((adv_r - img))+criterion(out_adv[-1], label)


        loss.backward()

        if apply_ti:
            adv.grad = gaussian_blur(adv.grad, kernel_size=(15, 15), sigma=(3, 3))

        if attack_type == 'mifgsm' or attack_type == 'dim':
            adv.grad = adv.grad / torch.mean(torch.abs(adv.grad), dim=(1, 2, 3), keepdim=True)
            adv_noise = adv_noise + adv.grad
        else:
            adv_noise = adv.grad

        # Optimization step
        if attack_type == 'pifgsm':
            amplification += alpha_beta * adv_noise.sign()
            cut_noise = torch.clamp(abs(amplification) - eps, 0, 10000.0) * torch.sign(amplification)
            projection = gamma * torch.sign(project_noise(cut_noise, stack_kern, padding_size))
            amplification += projection

            adv.data = adv.data + alpha_beta * adv_noise.sign() + projection
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
            adv.data.clamp_(0.0, 1.0)

        else:
            if index == 'pbsa':
                adv.data = adv.data + weight * step * adv_noise.sign()  # pma
            else:
                adv.data = adv.data + step * adv_noise.sign()
            # adv.data = adv.data + step * adv_noise.sign()  # pma
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)

            adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    mask = abs(adv - img)
    if skip and h is not None:
        h.remove()
    return adv.detach(), mask
