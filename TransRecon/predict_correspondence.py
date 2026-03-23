import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import logging
import argparse
import random
import numpy as np
import cv2 as cv
import torch
import os.path as osp
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.NetworkRunner import NetworkRunner

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


class Exper:
    def __init__(self, conf, exp_name='pred_correspondence', x_fov=None):
        self.device = torch.device('cuda')
        self.exp_name = exp_name
        self.conf = conf
        self.fov = x_fov

        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])

        self.network_runner = NetworkRunner(self.conf['run_network'])
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        self.network_runner.set_working_dir(
            osp.join(abs_dir, self.base_exp_dir, self.exp_name))

    def prepare_input(self):
        input_image_list = []
        background_list = []
        validmask_list = []

        for idx in range(len(self.dataset.images_lis)):
            img = self.dataset.image_at(idx)
            input_image_list.append(img)

        objmask_folder = osp.join(self.base_exp_dir, 'export_mask', 'mask')
        for idx in range(len(self.dataset.images_lis)):
            if os.path.exists(objmask_folder):
                mask = cv.imread(osp.join(objmask_folder,
                    '{}.png'.format(str(idx).zfill(3)))).astype(np.float32) / 255.0
                mask = mask[:, :, 0]
            else:
                mask = np.ones_like(input_image_list[idx][:, :, 0])
            validmask_list.append(mask)

        background_folder = osp.join(self.base_exp_dir, 'render_background', 'view')
        for idx in range(len(self.dataset.images_lis)):
            bkg = cv.imread(osp.join(background_folder,
                'background_{}.png'.format(str(idx)))).astype(np.float32) / 255.0
            background_list.append(bkg)

        self.network_runner.feed_input(input_image_list, background_list, validmask_list)

    def run_network(self):
        self.predict_correspondence, self.valid_mask_list = self.network_runner.run()

    def collect_result(self):
        img = self.dataset.image_at(0)
        self.original_img_size = img.shape[:2]  # [H, W]
        
        # Specify the background fov instead of the param in dataset        
        if self.fov is not None:
            self.dataset.set_xfov(self.fov)

        self.out_dir_list = []
        for idx in range(len(self.dataset.images_lis)):
            corres = self.predict_correspondence[idx]
            # Correspondence to 3D direction
            # The network outputs [0,1] normalized correspondence, which is resolution-invariant.
            # So we directly use original image size and camera intrinsics for unprojection,
            # regardless of whether the network ran at a different resolution.
            intrin_inv_mat = self.dataset.intrinsics_all_inv[idx, :3, :3].cpu().numpy()
            c2w_rot = self.dataset.pose_all[idx, :3, :3].cpu().numpy()
            corres = corres.reshape(-1, 2)
            p = np.stack([corres[:, 0] * self.original_img_size[1],
                          corres[:, 1] * self.original_img_size[0],
                          np.ones_like(corres[:, 0])], axis=1)
            p = intrin_inv_mat @ p.T
            rays_v = p / np.linalg.norm(p, axis=0, keepdims=True)
            rays_v = c2w_rot @ rays_v
            out_dir = rays_v.T.reshape(self.original_img_size[0], self.original_img_size[1], 3)
            self.out_dir_list.append(out_dir)

        out_dir = np.stack(self.out_dir_list, axis=0)
        np.save(os.path.join(self.base_exp_dir, self.exp_name, 'out_dir.npy'), out_dir)
        self.valid_mask_all = np.stack(self.valid_mask_list, axis=0)
        np.save(os.path.join(self.base_exp_dir, self.exp_name, 'valid_mask.npy'), self.valid_mask_all)

    def run(self):
        self.prepare_input()
        self.run_network()
        self.collect_result()


if __name__ == '__main__':
    print('Hello Ark')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/predict_corres.conf')
    parser.add_argument('--exp_name', type=str, default='pred_correspondence')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='cat')
    parser.add_argument('--x_fov', type=float, default=None)

    args = parser.parse_args()

    f = open(args.conf)
    conf_text = f.read()
    conf_text = conf_text.replace('CASE_NAME', args.case)
    f.close()

    conf = ConfigFactory.parse_string(conf_text)
    conf['dataset.data_dir'] = conf['dataset.data_dir'].replace('CASE_NAME', args.case)

    print("Deal case {}".format(args.case))
    torch.cuda.set_device(args.gpu)
    exper = Exper(conf, args.exp_name, args.x_fov)
    exper.run()
