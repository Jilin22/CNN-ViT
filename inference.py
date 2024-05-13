import os
import os.path as osp
import time

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from df_model import CNN_ViT


@torch.no_grad()
def intfc(fp: str):
    input = Image.open(fp)
    tensor_input = t(input).cuda().unsqueeze(0)
    flare_hat, output = model(tensor_input)
    output = t_inv(output.cpu().squeeze(0))
    flare_hat = t_inv(flare_hat.cpu().squeeze(0))
    return output, flare_hat


def check_dir(fp: str):
    if not osp.exists(fp):
        os.makedirs(fp)
        print(f"\t[MAKE] created out_dir: {fp}!")
    else:
        print(f"\t[SKIP] {fp} already exists!")


if __name__ == "__main__":
    model = CNN_ViT().cuda().eval()
    weight_path = "ckpts/network_latest.pth"
    model.load_state_dict(torch.load(weight_path)['params'])

    t = T.ToTensor()
    t_inv = T.ToPILImage()

    save_flare = 1
    test_path = "imgs/real"
    result_path = "imgs/deflare"

    check_dir(result_path)
    if save_flare:
        check_dir(osp.join(result_path, 'flare'))

    dt = []
    with torch.no_grad():
        for file in os.listdir(test_path):
            if osp.exists(osp.join(result_path, file)):
                print(f"\t[SKIP] {file} already exists!")
                continue
            img_path = osp.join(test_path, file)
            img_input = Image.open(img_path)

            tensor_input = t(img_input).cuda().unsqueeze(0)

            t_start = time.time()
            flare_hat, output = model(tensor_input)
            dt.append(time.time() - t_start)

            deflare = t_inv(output.squeeze())
            deflare.save(osp.join(result_path, file))
            if save_flare:
                t_inv(flare_hat.squeeze()).save(osp.join(result_path, 'flare', file))

    dt.pop(0)
    info = f"\t[{len(os.listdir(test_path))}] imgs in total, average time: {np.mean(dt) * 1000:0.2f} ms."
    print(info)

    print("INFO: done!")
