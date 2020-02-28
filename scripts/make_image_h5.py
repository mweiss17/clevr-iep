import os
import h5py
import numpy as np
from scipy.misc import imread, imresize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_image_dir', required=True)
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_h5_file', required=True)

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='resnet101')
parser.add_argument('--model_stage', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--multi_dir', action='store_true')



def main(args):
    input_paths = []
    for fn in os.listdir(args.input_image_dir):
        if not fn.endswith('.png') and not fn.endswith('.jpg'): continue
        idx = os.path.join(os.path.splitext(fn)[0].split('_')[-1])
        input_paths.append((os.path.join(args.input_image_dir, fn), idx))

    img_size = (args.image_height, args.image_width)
    with h5py.File(args.output_h5_file, 'w') as f:
        im_dset = f.create_dataset('images', (len(input_paths), 3, args.image_height, args.image_width),
                                   dtype=np.float32)

        for i, (path, idx) in enumerate(input_paths):
            img = imread(path, mode='RGB')
            img = imresize(img, img_size, interp='bicubic')
            img = img.transpose(2, 0, 1)[None]
            im_dset[i] = img

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
