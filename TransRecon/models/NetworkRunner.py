import cv2
import numpy as np
import os
import subprocess
import flow_vis
from matplotlib import cm


def decode_corres(path):
    # Read .flo format correspondence file
    # Similar to optical flow format, but stores correspondence as np.float32
    TAG_FLOAT = 202021.25
    with open(path, 'rb') as file:
        flag = np.fromfile(file, dtype=np.float32, count=1)
        if flag != TAG_FLOAT:
            raise Exception('unable to read %s, maybe broken' % path)
        size = np.fromfile(file, dtype=np.int32, count=2)
        flo = np.fromfile(file, dtype=np.float32).reshape([size[1], size[0], 2])
    return flo


def colormap(diff, thres):
    diff_norm = np.clip(diff, 0, thres) / thres
    diff_cm = cm.jet(diff_norm)[:, :, :3]
    return diff_cm[:, :, ::-1]


class NetworkRunner():
    def __init__(self, conf):
        self.run_network_dir = "../RCEstimate"
        self.run_network_cmd = "python test_recon.py --input_dir {} --output_dir {} --opt configs/test_recon.yaml"
        self.conf = conf
        self.input_resolution = np.array(self.conf.get_list('input_resolution', default=[512, 512]))
        self.rgb_threshold = self.conf.get_float('rgb_threshold', default=0.1)

    def feed_input(self, input_img_list, background_list, valid_mask_list):
        assert len(input_img_list) == len(background_list) == len(valid_mask_list)
        assert len(valid_mask_list[0].shape) == 2
        self.img_num = len(input_img_list)
        self.input_img_list = input_img_list
        self.background_list = background_list
        self.valid_mask_list = valid_mask_list
        H, W = self.input_img_list[0].shape[:2]
        self.original_img_size = (H, W)

    def resize_to_net(self, img):
        """Resize image to network input resolution."""
        return cv2.resize(img, (self.input_resolution[1], self.input_resolution[0]))

    def resize_to_original(self, img):
        """Resize image back to original resolution."""
        original_h, original_w = self.original_img_size
        return cv2.resize(img, (original_w, original_h))

    def set_working_dir(self, working_dir):
        self.working_dir = working_dir

    def prepare_input(self):
        self.net_input_dir = os.path.join(self.working_dir, 'input_data')
        self.raw_output_dir = os.path.join(self.working_dir, 'raw_output_data')
        self.clean_output_dir = os.path.join(self.working_dir, 'clean_output_data')
        os.makedirs(self.net_input_dir, exist_ok=True)
        os.makedirs(self.raw_output_dir, exist_ok=True)
        os.makedirs(self.clean_output_dir, exist_ok=True)

        for idx in range(self.img_num):
            input_img = self.resize_to_net(self.input_img_list[idx])
            input_img = (255 * input_img).clip(0, 255).astype(np.uint8)
            background = self.resize_to_net(self.background_list[idx])
            background = (255 * background).clip(0, 255).astype(np.uint8)
            valid_mask = self.resize_to_net(self.valid_mask_list[idx])
            valid_mask = (valid_mask > 0.5).astype(np.uint8) * 255

            cv2.imwrite(os.path.join(self.net_input_dir, 'input_{}.png'.format(str(idx).zfill(3))), input_img)
            cv2.imwrite(os.path.join(self.net_input_dir, 'background_{}.png'.format(str(idx).zfill(3))), background)
            cv2.imwrite(os.path.join(self.net_input_dir, 'mask_{}.png'.format(str(idx).zfill(3))), valid_mask)

    def run_network(self):
        subprocess.run(self.run_network_cmd.format(self.net_input_dir, self.raw_output_dir),
                       shell=True, cwd=self.run_network_dir)

    def collect_result(self):
        self.basename_list = []
        for idx in range(self.img_num):
            self.basename_list.append(str(idx).zfill(3))

        valid_mask_list = []
        self.pred_corres_list = []
        for idx, basename in enumerate(self.basename_list):
            raw_corres = decode_corres(os.path.join(self.raw_output_dir, '{}_pred_corres.flo'.format(basename)))
            # Correspondence is in [0,1] normalized coords, spatial interpolation preserves the values
            pred_corres = self.resize_to_original(raw_corres)
            self.pred_corres_list.append(pred_corres)
            pred_corres_map = flow_vis.flow_to_color(pred_corres * 2 - 1)
            visual_img = pred_corres_map.astype(np.uint8)
            visual_img = visual_img[:, :, ::-1]

            # Compute valid mask using RGB threshold:
            # compare the network's refracted image with the original input
            network_result = cv2.imread(os.path.join(self.raw_output_dir, 'visualization--{}.png'.format(basename))) / 255.0
            refract_img = network_result[:, :self.input_resolution[1], :]
            refract_img = self.resize_to_original(refract_img)
            input_img = self.input_img_list[idx]
            img_error = np.abs(refract_img - input_img).mean(axis=-1)
            error_mask = img_error < self.rgb_threshold
            valid_mask = self.valid_mask_list[idx]
            valid_mask = valid_mask * error_mask
            valid_mask_list.append(valid_mask)
            visual_img = visual_img * valid_mask[:, :, None]

            cv2.imwrite(os.path.join(self.clean_output_dir, f'visual_{basename}.png'), visual_img)
        self.valid_mask_list = valid_mask_list

    def run(self):
        self.prepare_input()
        self.run_network()
        self.collect_result()
        return self.pred_corres_list, self.valid_mask_list
