from networks.decoders.fpn import FPNSegmentationHead
from networks.decoders.diffusion  import DiffSeg


def build_decoder(name, **kwargs):

    if name == 'fpn':
        return FPNSegmentationHead(**kwargs)
    elif name == "diffusion":
        return DiffSeg(**kwargs)
    else:
        raise NotImplementedError
