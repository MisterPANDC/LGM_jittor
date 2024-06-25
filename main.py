import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import argparse
import jittor as jt
import numpy as np

from options import config_defaults
from models import LGM

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='lrm', choices=config_defaults.keys())
parser.add_argument('--checkpoint', type=str, default='checkpoints/lrm.pth')
parser.add_argument('--image_path', type=str, default='images')
parser.add_argument('--output', type=str, default='output')
args = parser.parse_args()

opt = config_defaults[args.config]
# print(opt)
model = LGM(opt)
if os.path.exists(args.checkpoint):
    model.load_state_dict(jt.load(args.checkpoint))
else:
    print(f'checkpoint not found: {args.checkpoint}')

model = model.cuda().eval()

rays_embeddings = model.prepare_default_rays('cuda')

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = jt.zeros((4, 4), dtype=jt.float32).to("cuda")
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

def main():
    name = os.path.splitext(os.path.basename(args.output))[0]
    print(f'[INFO] Processing {args.output} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)

    # read input image
    image = cv2.imread(args.image_path)
    image = jt.array(image).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    image = jt.nn.interpolate(image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    image = jt.concat([image, rays_embeddings], dim=1).unsqueeze(0)

    gaussians = model.forward_gaussians(image)

    # save gaussians
    model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '.ply'))

if __name__ == '__main__':
    main()
