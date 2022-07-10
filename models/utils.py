import torch

from models import network

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(GNet_opt):
    # Initialize the network
    # generator = network.Mainstream(GNet_opt.args)
    generator = getattr(network, GNet_opt.name)(GNet_opt.args)

    # Init or Load value for the network
    network.weights_init(generator, init_type=GNet_opt.init_type, init_gain=GNet_opt.init_gain)
    print('Generator is created!')
    if GNet_opt.finetune_path != "":
        # pretrained_net = torch.load(GNet_opt.finetune_path)
        # generator = load_dict(generator, pretrained_net)
        generator.load_ckpt(GNet_opt.finetune_path, force_load=hasattr(GNet_opt, 'force_load') and GNet_opt.force_load)
        print('Generator is loaded!')
    return generator

def create_generator_val(GNet_opt, model_path=None, force_load=False):
    # Initialize the network
    # generator = network.Mainstream(GNet_opt.args)
    generator = getattr(network, GNet_opt.name)(GNet_opt.args)
    # Init or Load value for the network
    
    network.weights_init(generator, init_type=GNet_opt.init_type, init_gain=GNet_opt.init_gain)
    print('Generator is created!')

    if model_path is not None:
        generator.load_ckpt(model_path, force_load=force_load)
        print('Generator is loaded!')
    return generator

def create_discriminator(DNet_opt):
    # Initialize the network
    # discriminator = network.PatchDiscriminator70(DNet_opt.args)
    discriminator = getattr(network, DNet_opt.name)(DNet_opt.args)
    # Init the network
    network.weights_init(discriminator, init_type=DNet_opt.init_type, init_gain=DNet_opt.init_gain)
    print('Discriminators is created!')
    return discriminator

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net
