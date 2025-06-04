from .csdnet import CSDNet


def build_model(args):
    assert args.model_type in ['ResNet']
    if args.model_type == 'ResNet':
        return CSDNet(args)
