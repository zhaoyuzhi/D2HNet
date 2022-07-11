import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class Parallel(object):
    def __init__(self, num_gpus, rank = None, world_size = None):
        # set device
        if num_gpus <= 0:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cpu")
            # cudnn open
            torch.backends.cudnn.enabled = False

        self.device = device
        self.num_gpus = num_gpus

        self.gpu_ids = range(num_gpus) if num_gpus > 0 else []

        if rank is not None:
            n = num_gpus // world_size
            self.device_ids = list(range(rank * n, (rank + 1) * n))

    def wrapper(self, entry):
        if len(self.gpu_ids) <= 1:
            # entry = nn.DataParallel(entry, self.gpu_ids)
            entry = entry.to(self.device)
        else:
            # if not isinstance(entry, nn.DataParallel):
                # print('warp net')
            entry = entry.to(self.device)
            entry = nn.DataParallel(entry, self.gpu_ids)
        return entry

    def wrapper_dist(self, entry):
        entry = entry.to(self.device_ids[0])
        return DDP(entry, device_ids = self.device_ids)

    def cleanup(self):
        dist.destroy_process_group()
