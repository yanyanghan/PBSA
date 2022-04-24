import argparse
import datetime
import json
import logging
import os
import random
import torch
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision import transforms, models, utils as vutils
from tqdm import tqdm

import vit_models
from attack import normalize, local_adv
from dataset import AdvImageNet
import pretrainedmodels

targeted_class_dict = {
    24: "Great Grey Owl",
    99: "Goose",
    245: "French Bulldog",
    344: "Hippopotamus",
    471: "Cannon",
    555: "Fire Engine",
    661: "Model T",
    701: "Parachute",
    802: "Snowmobile",
    919: "Street Sign ",
}
# GPU
gpus = "4,6"
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Transformers')
    parser.add_argument('--test_dir', default='data/train', help='ImageNet Validation Data')
    parser.add_argument('--dataset', default="imagenet_5k", help='dataset name')
    parser.add_argument('--src_model', type=str, default='deit_small_patch16_224', help='Source Model Name')
    parser.add_argument('--src_pretrained', type=str, default=None, help='pretrained path for source model')
    parser.add_argument('--scale_size', type=int, default=256, help='')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--eps', type=int, default=8, help='Perturbation Budget')
    parser.add_argument('--iter', type=int, default=10, help='Attack iterations')
    parser.add_argument('--index', type=str, default='last', help='last or all,pma')
    parser.add_argument('--attack_type', type=str, default='pgd', help='fgsm, mifgsm, dim, pgd, pbsa')
    parser.add_argument('--tar_ensemble', action="store_true", default=False)
    parser.add_argument('--apply_ti', action="store_true", default=False)
    parser.add_argument('--save_im', action="store_true", default=False)
    parser.add_argument('--skip', action="store_true", default=False, help='SGM')
    parser.add_argument('--m', type=int, default=5, help='Mask Number')
    parser.add_argument('--k', type=int, default=3, help='Number of encoders selected')
    parser.add_argument('--p', type=float, default=0.3, help='binomial probability')

    parser.add_argument('--tar_model', type=list,
                        default=["T2t_vit_24", "tnt_s_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224",
                                 "T2t_vit_7", "resnet50", "resnet152", "vgg19", "densenet201", "senet154"],
                        help='Target Model Name')
    return parser.parse_args()


def get_model(model_name):
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    other_model_names = vars(vit_models)
    # get the source model
    if model_name in model_names:
        model = models.__dict__[model_name](pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'deit' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'hierarchical' in model_name or "ensemble" in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'vit' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'T2t' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'tnt' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'swin' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    return model, mean, std


#  Test Samples
def get_data_loader(args, verbose=True):
    data_transform = transforms.Compose([
        transforms.Resize(args.scale_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
    ])

    test_dir = args.test_dir
    if args.dataset == "imagenet_1k":
        test_set = AdvImageNet(image_list="data/image_list_1k.json", root=test_dir, transform=data_transform)
    else:
        test_set = AdvImageNet(root=test_dir, transform=data_transform)
    test_size = len(test_set)
    if verbose:
        print('Test data size:', test_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                              pin_memory=True)
    return test_loader, test_size


def main():
    # setup run
    args = parse_args()
    args.exp = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{args.index}_{random.randint(1, 100)}"
    os.makedirs(f"report/{args.exp}")
    json.dump(vars(args), open(f"report/{args.exp}/config.json", "w"), indent=4)  # 编码json文件

    logger = logging.getLogger(__name__)
    logfile = f'report/{args.exp}/{args.src_model}_5k_val_eps_{args.eps}.log'
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)

    # load source and target models
    src_model, src_mean, src_std = get_model(args.src_model)
    if args.src_pretrained is not None:
        if args.src_pretrained.startswith("https://"):
            src_checkpoint = torch.hub.load_state_dict_from_url(args.src_pretrained, map_location='cpu')
        else:
            src_checkpoint = torch.load(args.src_pretrained, map_location='cpu')
        src_model.load_state_dict(src_checkpoint['model'])
    src_model = torch.nn.DataParallel(src_model)
    src_model = src_model.to(device)
    src_model.eval()

    n_models = len(args.tar_model)
    # Setup-Data
    test_loader, test_size = get_data_loader(args)

    # setup eval parameters
    eps = args.eps / 255
    criterion = torch.nn.CrossEntropyLoss()
    logger.info(f'Source: "{args.src_model}" \t Target: "{args.tar_model}" \t Eps: {args.eps} \t Index: {args.index} \t'
                f'Mask num: "{args.m}" \t encoder_num: "{args.k}" \t bern_prob: "{args.p}" \t skip: "{args.skip}" \t'
                f'\t Attack: {args.attack_type}')
    avg_fooling = [0.0 for _ in range(n_models)]
    test_number = 5
    for mm in range(test_number):
        clean_acc = [0.0 for _ in range(n_models)]
        adv_acc = [0.0 for _ in range(n_models)]
        fool_rate = [0.0 for _ in range(n_models)]
        with tqdm(enumerate(test_loader), total=len(test_loader)) as p_bar:
            for i, (img, label) in p_bar:
                img, label = img.to(device), label.to(device)

                adv, mask = local_adv(src_model, criterion, img, label, eps, attack_type=args.attack_type, iters=args.iter,
                            std=src_std, mean=src_mean, index=args.index, m=args.m, k=args.k, p=args.p, apply_ti=args.apply_ti, skip=args.skip)
                for idx in range(n_models):
                    tar_model, tar_mean, tar_std = get_model(args.tar_model[idx])
                    tar_model = torch.nn.DataParallel(tar_model)
                    tar_model = tar_model.to(device)
                    tar_model.eval()
                    with torch.no_grad():
                        if args.tar_ensemble:
                            clean_out = tar_model(normalize(img.clone(), mean=tar_mean, std=tar_std), get_average=True)
                        else:
                            clean_out = tar_model(normalize(img.clone(), mean=tar_mean, std=tar_std))
                    if isinstance(clean_out, list):
                        clean_out = clean_out[-1].detach()

                    clean_acc[idx] += torch.sum(clean_out.argmax(dim=-1) == label).item()
                    with torch.no_grad():
                        if args.tar_ensemble:
                            adv_out = tar_model(normalize(adv.clone(), mean=tar_mean, std=tar_std), get_average=True)
                        else:
                            adv_out = tar_model(normalize(adv.clone(), mean=tar_mean, std=tar_std))
                    if isinstance(adv_out, list):
                        adv_out = adv_out[-1].detach()
                    adv_acc[idx] += torch.sum(adv_out.argmax(dim=-1) == label).item()
                    fool_rate[idx] += torch.sum(adv_out.argmax(dim=-1) != clean_out.argmax(dim=-1)).item()

                if args.save_im:
                    for m in range(125):
                        vutils.save_image(vutils.make_grid(adv[m], normalize=False, scale_each=True),
                                              'adv_{}.png'.format(label[m]))
                        vutils.save_image(vutils.make_grid(img[m], normalize=False, scale_each=True),
                                              'img_{}.png'.format(label[m]))

        for idx in range(n_models):
            avg_fooling[idx] += fool_rate[idx] / test_size
            print('Model:{}\t Clean:{:.3%}\t Adv :{:.3%}\t Fooling Rate:{:.3%}'.format(idx, clean_acc[idx] / test_size, adv_acc[idx] / test_size,
                                                                           fool_rate[idx] / test_size))
            logger.info(
                'Eps:{0} \t Clean:{1:.3%} \t Adv :{2:.3%}\t Fooling Rate:{3:.3%}'.format(int(eps * 255),
                                                                                         clean_acc[idx] / test_size,
                                                                                         adv_acc[idx] / test_size,
                                                                                         fool_rate[idx] / test_size))
            json.dump({"eps": int(eps * 255),
                       "clean": clean_acc[idx] / test_size,
                       "adv": adv_acc[idx] / test_size,
                       "fool rate": fool_rate[idx] / test_size, },
                      open(f"report/{args.exp}/results.json", "w"), indent=4)

    for idx in range(n_models):
        print('Model:{}\t Fooling Rate:{:.3%}'.format(idx, avg_fooling[idx] / test_number))


if __name__ == '__main__':
    main()
