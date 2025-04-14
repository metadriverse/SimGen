import sys
if './' not in sys.path:
	sys.path.append('./')

import torch

from utils.share import *
from models.util import create_model


def init_local(sd_weights_path, config_path, output_path):
    pretrained_weights = torch.load(sd_weights_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']
    model = create_model(config_path=config_path)
    scratch_dict = model.state_dict()
    target_dict = {}
    for sk in scratch_dict.keys():
        if sk.replace('local_adapter.', 'model.diffusion_model.') in pretrained_weights.keys():
            target_dict[sk] = pretrained_weights[sk.replace('local_adapter.', 'model.diffusion_model.')].clone()
        else:
            target_dict[sk] = scratch_dict[sk].clone()
            print('new params: {}'.format(sk))
    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)
    print('Done.')


if __name__ == '__main__':
    mode = sys.argv[1]
    assert mode in ['init_local']
    if mode == 'init_local':
        assert len(sys.argv) == 5
        init_local(sys.argv[2], sys.argv[3], sys.argv[4])