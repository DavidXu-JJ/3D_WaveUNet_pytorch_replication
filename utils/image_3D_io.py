import os, cv2, sys
import numpy as np
from datetime import datetime


from constant import Image2DName_Length
IMAGE_SUFFIXES = ['.jpg', '.jpeg', '.tiff', '.bmp', '.png', '.tif']

def load_image_3d(image_root):
    image_name_list = os.listdir(image_root)
    image_name_list.sort()
    image_3d = []
    for image_name in image_name_list:
        _, suffix = os.path.splitext(image_name)
        if suffix not in IMAGE_SUFFIXES:
            continue
        # 2D array
        image = cv2.imread(os.path.join(image_root, image_name), 0)
        image_3d.append(image)
    # z, y, x
    return np.array(image_3d)

def save_iamge_3d(image_3d, image_save_root, dim = 0, suffix = '.tiff'):
    assert dim < len(image_3d.shape)
    assert suffix in IMAGE_SUFFIXES
    if not os.path.isdir(image_save_root):
        os.makedirs(image_save_root)

    if len(os.listdir(image_save_root)):
        os.system('rm '+ os.path.join(image_save_root,'*'))
    depth = image_3d.shape[dim]
    for index in range(depth):
        image_full_name = os.path.join(image_save_root, str(index).zfill(Image2DName_Length) + suffix)
        flag = cv2.imwrite(image_full_name, image_3d[index]) # ?

        if flag == False:
            print('{} to save {}-th image in {}'.format(flag, index, image_save_root))
            break

        time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
        sys.stdout.write('\r{}-------- saving {} / {} 2D image ...'.format(time, index, depth))
        sys.stdout.flush()
    time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
    sys.stdout.write('\n{}-------- finish saving !\n'.format(time))