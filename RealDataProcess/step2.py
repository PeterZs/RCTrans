# process the colmap camera params to the unit sphere
import trimesh
import numpy as np
import os
import cv2
import argparse


class RealData():
    def __init__(self, project_dir,downsample_size=[512,512]):
        self.project_dir = project_dir
        self.downsample_size = downsample_size
        
        self.objpoint_path = f'{project_dir}/object_point_cloud.ply'
        self.camerainfo_path = f'{project_dir}/cameras.txt'
        self.imginfo_path = f'{project_dir}/images.txt'
        self.camera_dict = {}
        self.raw_camera_dict = {}
        self.images_dir = f'{project_dir}/images'
        self.id_to_imgname = {}

            
    def run(self):
        self.load_and_process_colmap_cam()
        self._normalize()
        self.downsample_image()
        self.save()
    
    def downsample_image(self):
        # downsample images

        self.original_img_size = None
        self.downsample_image_dir = f'{self.project_dir}/image'
        downsample_size = self.downsample_size[::-1]
        os.makedirs(self.downsample_image_dir, exist_ok=True)
        for idx in range(self.img_num):
            img_path = os.path.join(self.images_dir, self.id_to_imgname[idx])
            img = cv2.imread(img_path)
            if self.original_img_size is None:
                self.original_img_size = img.shape[:2]
            img = cv2.resize(img, downsample_size)
            cv2.imwrite(os.path.join(self.downsample_image_dir, '{}.png'.format(str(idx+1).zfill(3))), img)

        self.downsample_camera_intrinsics()
    
    def downsample_camera_intrinsics(self):
        # update camera intrinsic params
        
        # I'm not sure if the following code is correct when the image ratio is changed
        # Strongly recommend to keep the original image ratio
        
        h_factor = self.downsample_size[0] / self.original_img_size[0]
        w_factor = self.downsample_size[1] / self.original_img_size[1]
        intri_mat = self.camera_dict['intrinsic_mat']
        intri_mat[0,2] *= w_factor
        intri_mat[1,2] *= h_factor
        intri_mat[0,0] *= w_factor
        intri_mat[1,1] *= h_factor
        self.camera_dict['intrinsic_mat'] = intri_mat
        for idx in range(self.img_num):
            rot_mat = self.camera_dict['rot_mat_{}'.format(idx)]
            t = self.camera_dict['t_{}'.format(idx)]
            extri_mat = np.concatenate((rot_mat, t.reshape(3,1)), axis=1)
            self.camera_dict['world_mat_{}'.format(idx)] = np.dot(intri_mat, extri_mat)
    
    def quate2rotmat(self,Q):
        # q = a + bi + cj + dk
        a = float(Q[0])
        b = float(Q[1])
        c = float(Q[2])
        d = float(Q[3])

        R = np.array([[2*a**2-1+2*b**2, 2*b*c+2*a*d,     2*b*d-2*a*c],
                    [2*b*c-2*a*d,     2*a**2-1+2*c**2, 2*c*d+2*a*b],
                    [2*b*d+2*a*c,     2*c*d-2*a*b,     2*a**2-1+2*d**2]])
        return np.transpose(R)    
    
    def _compute_rotation(self,vert, forward):
        y = np.cross(vert, forward)
        x = np.cross(y, vert)

        vert = vert/np.linalg.norm(vert)
        x = x/np.linalg.norm(x)
        y = y/np.linalg.norm(y)
        R = np.stack([x, y, vert], 0)
        return R
    
    def load_and_process_colmap_cam(self):
        c = 0
        with open(self.camerainfo_path, 'r') as camParams:
            for p in camParams.readlines():
                c += 1
                if c <= 3: # skip comments
                    continue
                else:
                    line = p.strip().split(' ')
                    imgW, imgH = int(line[2]), int(line[3])
                    f = float(line[4])
                    cxp, cyp = int(float(line[5])), int(float(line[6]))
                    intri_mat = np.eye(3)
                    intri_mat[0,0] = f
                    intri_mat[1,1] = f
                    intri_mat[0,2] = cxp
                    intri_mat[1,2] = cyp
                    break
        self.raw_camera_dict['intrinsic_mat'] = intri_mat
        
        c = 0
        idx = 0 
        scale_mat = np.eye(4)
        with open(self.imginfo_path, 'r') as camPoses:
            for cam in camPoses.readlines():
                c += 1
                if c <= 3: # skip comments
                    continue
                elif c == 4:
                    numImg = int(cam.strip().split(',')[0].split(':')[1])
                    print('Number of images:', numImg)
                    self.img_num = numImg
                    
                else:
                    if c % 2 == 1:
                        line = cam.strip().split(' ')
                        ori_rot_mat = self.quate2rotmat(line[1:5])
                        ori_t = np.array([float(line[5]), float(line[6]), float(line[7])])
                        
                        ori_extri_mat = np.concatenate((ori_rot_mat, ori_t.reshape(3,1)), axis=1)
                        ori_extri_mat = np.concatenate([ori_extri_mat, np.array([[0,0,0,1]])], axis=0)
                        extri_mat = ori_extri_mat
                        
                        rot_mat = extri_mat[:3,:3]
                        t = extri_mat[:3,3]
                        
                        origin = -np.dot(np.transpose(rot_mat),t.reshape(3,1)).reshape(3)
                        dir = np.dot(np.transpose(rot_mat),np.array([0,0,1]).reshape(3,1)).reshape(3)
                        
                        self.raw_camera_dict['rot_mat_{}'.format(idx)] = rot_mat                
                        self.raw_camera_dict['origin_{}'.format(idx)] = origin
                        self.raw_camera_dict['t_{}'.format(idx)] = t.reshape(3,1)
                        self.raw_camera_dict['scale_mat_{}'.format(idx)] = scale_mat
                        self.id_to_imgname[idx] = line[-1]
                        
                        idx += 1
    
    
    def _normalize(self):
        self.ref_points = trimesh.load(self.objpoint_path).vertices
        self.ref_points = np.array(self.ref_points)
        max_pt, min_pt = np.max(self.ref_points, 0), np.min(self.ref_points, 0)
        center = (max_pt + min_pt) * 0.5
        offset = -center.reshape(3,1) # x1 = x0 + offset
        scale = 1 / np.max(np.linalg.norm(self.ref_points - center[None,:], 2, 1)) # x2 = scale * x1
        scale *= 0.9
        directions = np.loadtxt(f'{self.project_dir}/axis.txt')
        up = directions[0]
        forward = directions[1]
        z_point = directions[-1]
        up, forward = up/np.linalg.norm(up), forward/np.linalg.norm(forward)
        R_rec = self._compute_rotation(up, forward) # x3 = R_rec @ x2
        self.ref_points = scale * (self.ref_points + offset.reshape(3)) @ R_rec.T
        self.z_point = scale * (z_point + offset.reshape(3)) @ R_rec.T
        self.scale_rect = scale
        self.offset_rect = offset
        self.R_rect = R_rec
        intri_mat = self.raw_camera_dict['intrinsic_mat']
        self.camera_dict['intrinsic_mat'] = intri_mat
        for idx in range(self.img_num):
            R = self.raw_camera_dict['rot_mat_{}'.format(idx)]
            t = self.raw_camera_dict['t_{}'.format(idx)]
            R_new = R @ R_rec.T
            t_new = (t - R @ offset) * scale
            self.camera_dict['rot_mat_{}'.format(idx)] = R_new
            self.camera_dict['t_{}'.format(idx)] = t_new
            self.camera_dict['scale_mat_{}'.format(idx)] = self.raw_camera_dict['scale_mat_{}'.format(idx)]
            origin = -np.dot(np.transpose(R_new),t_new.reshape(3,1)).reshape(3)
            self.camera_dict['origin_{}'.format(idx)] = origin
            extri_mat = np.concatenate((R_new, t_new.reshape(3,1)), axis=1)
            self.camera_dict['world_mat_{}'.format(idx)] = np.dot(intri_mat, extri_mat)
    
    def save(self):
        np.savez(os.path.join(self.project_dir,'cameras_sphere.npz'), **self.camera_dict)
        np.savez(os.path.join(self.project_dir,'object_sphere.npz'), **self.camera_dict)
        with open(os.path.join(self.project_dir,'min_z.txt'), 'w') as f:
            f.write(str(np.round(self.z_point[-1] + 0.01, 2)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True)
    parser.add_argument('--downsample_size', type=int, nargs=2, default=[512,512])
    
    args = parser.parse_args()
    data = RealData(args.project_dir,downsample_size=args.downsample_size)
    data.run()


if __name__ == '__main__':
    main()
