import os
import numpy as np
from PIL import Image
import sys


class Results(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _read_mask(self, sequence, frame_id):
        try:
            mask_path = os.path.join(self.root_dir, sequence, f'{frame_id}.png')
            return np.array(Image.open(mask_path))
        except IOError as err:
            print(os.path.join(self.root_dir, sequence, f'{frame_id}.png'))
            sys.stdout.write(sequence + " frame %s not found!\n" % frame_id)
            sys.stdout.write("The frames have to be indexed PNG files placed inside the corespondent sequence "
                             "folder.\nThe indexes have to match with the initial frame.\n")
            sys.stderr.write("IOError: " + err.strerror + "\n")
            print("IOError: " + err.strerror + "\n")
            sys.exit()

    def read_masks(self, sequence, masks_id):
        # print(f"{sequence} DONE 1.91")
        mask_0 = self._read_mask(sequence, masks_id[0])
        # print(f"{sequence} DONE 1.92")
        masks = np.zeros((len(masks_id), *mask_0.shape), dtype=mask_0.dtype)
        # print(f"{sequence} DONE 1.93")
        for ii, m in enumerate(masks_id):
            masks[ii, ...] = self._read_mask(sequence, m)
        # print(f"{sequence} DONE 1.94")
        masks = np.where(masks == 255, 0, masks)
        # print(f"{sequence} DONE 1.95")
        num_objects = int(np.max(masks))
        # print(f"{sequence} DONE 1.96")
        tmp = np.ones((num_objects, *masks.shape), dtype=masks.dtype)
        # print(f"{sequence} DONE 1.97")
        tmp = tmp * np.arange(1, num_objects + 1, dtype=tmp.dtype)[:, None, None, None]
        # print(f"{sequence} DONE 1.98")
        masks = (tmp == masks[None, ...]) > 0
        # print(f"{sequence} DONE 1.99")
        return masks

    def get_sequences(self):
        return os.listdir(self.root_dir)