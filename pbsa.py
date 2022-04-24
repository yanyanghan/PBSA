import numpy as np
import torch

def norm(grad):
    square = np.sum(np.square(grad.cpu().numpy()), axis=(1, 2, 3), keepdims=True)
    nor_grad = (grad.cpu().numpy()) / np.sqrt(square)
    return torch.from_numpy(nor_grad).cuda()


def normalize(t, mean, std):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


def att_grad(model, img_tmp, criterion, label, p, mean, std):
    mask = torch.from_numpy(np.random.binomial(1, p, img_tmp.size())).cuda()
    img = img_tmp * mask
    img.retain_grad()
    out, patch_a_ma = model(normalize(img, mean, std))
    patch_a_ma = torch.stack(patch_a_ma, dim=0)
    loss = criterion(out[-1], label)
    loss.backward(retain_graph=True)
    w1_x = (img.grad).clone()
    img.grad.data.zero_()
    weight = []
    for layer in range(12):
        retain_graph = True
        if layer == 11:
            retain_graph = False
        patch_a_ma[layer].backward(torch.as_tensor(patch_a_ma[layer]).cuda(), retain_graph)
        w2_x = (img.grad).clone()
        a = 1e-6*torch.ones(w2_x.size()).cuda()
        w2_x = torch.where(w2_x == 0, a, w2_x)
        w = w1_x / w2_x
        weight.append(w)
        img.grad.data.zero_()
    weight = torch.stack(weight, dim=0)
    return weight


def pma(img_tmp, model, criterion, label, mean, std, mask_num, p, k):
    '''
    :param img_tmp: 原始图像
    :param model: 源模型
    :param mask_num: 需要生成的mask的数量
    :param p: 生成mask的概率
    :param k: 输出的有用encoder的数量
    :return: 平均注意力map梯度
    '''
    img = img_tmp.detach()
    img.requires_grad = True
    for l in range(mask_num):
        weight = att_grad(model, img, criterion, label, p=1-p, mean=mean, std=std)
        if l == 0:
            weights = weight
        else:
            weights += weight
    avg_a_ma = weights / mask_num
    a_ma = att_grad(model, img, criterion, label, p=1, mean=mean, std=std)
    diff = torch.abs(a_ma-avg_a_ma)
    for dim in range(3):
        diff = torch.sum(diff, dim=1)
    diff = torch.sum(diff, dim=1, keepdim=True).transpose(0, 1)
    index = torch.sort(diff, descending=False, dim=-1)[1]
    norm_a_ma = 1 + norm(avg_a_ma)
    one = torch.ones(norm_a_ma.size()).cuda()
    norm_a_ma = torch.where(norm_a_ma>1, norm_a_ma, one)
    a = norm_a_ma[torch.squeeze(index)[0].item()]
    return index[:, :k], a


def backward_hook(gamma):
    def _backward_hook(module, grad_in, grad_out):
        return (gamma * grad_in[0], )
    return _backward_hook


def register_hook_for_deit(model, layer, gamma):
    backward_hook_sgm = backward_hook(gamma)
    sub_model = (model.module).blocks
    for name, module in sub_model.named_modules():
        if 'attn.attn_drop' in name and not name.split('.')[0] in layer:  # pma
            module.register_backward_hook(backward_hook_sgm)


def register_hook_for_all(model, gamma):
    backward_hook_sgm = backward_hook(gamma)
    sub_model = (model.module).blocks
    for name, module in sub_model.named_modules():
        if 'attn.attn_drop' in name:  # all
            module.register_backward_hook(backward_hook_sgm)