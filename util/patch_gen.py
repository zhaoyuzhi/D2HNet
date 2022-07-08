import numpy as np

class PatchGenerator(object):

    def __init__(self, H, W, patch_size=None, padding=16):
        # assert H == W and H % 2 == 0
        self.H = H
        self.W = W
        self.padding = padding
        self.patch_size = self._calc_patch_size(patch_size)
    
    def _calc_patch_size(self, patch_size):
        if patch_size is None:
            assert self.padding is not None
            patch_size = self.H // 2 + self.padding
            return patch_size  
        else:
            return patch_size

    def next_patch(self):
        H, W = self.H, self.W
        padding = self.padding
        patch_size = self.patch_size
        H_block_num = int(np.ceil((H - padding * 2) / (patch_size - padding * 2)))
        W_block_num = int(np.ceil((W - padding * 2) / (patch_size - padding * 2)))


        for i in range(H_block_num):
            h = i * (patch_size - 2 * padding)
            if i == 0:
                h = 0
            elif i == H_block_num - 1:
                h = H - patch_size
            
            top_padding, bottom_padding = padding, padding
            if i == 0:
                top_padding = 0
            elif i == H_block_num - 1:
                bottom_padding = 0
            
            for j in range(W_block_num):
                w = j * (patch_size - 2 * padding)
                if j == 0:
                    w = 0
                elif j == W_block_num - 1:
                    w = W - patch_size
                left_padding, right_padding = padding, padding
                if j == 0:
                    left_padding = 0
                elif j == W_block_num - 1:
                    right_padding = 0
                
                yield h, w, top_padding, left_padding, bottom_padding, right_padding


