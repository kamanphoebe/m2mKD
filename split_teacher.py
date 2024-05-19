import os
import argparse
import torch
from deep_incubation.timm.models import create_model


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--save_dir', default='./log_dir/FT_large', type=str)
    parser.add_argument('--ckpt_path', default='./log_dir/FT_large/finished_checkpoint.pth', type=str)
    parser.add_argument('--model', default='vit_large_patch16_224', type=str)
    parser.add_argument('--divided_depths', default=[6, 6, 6, 6], nargs='+', type=int)
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Split teacher model pretrained by Deep Incubation', parents=[get_args_parser()])
    args = parser.parse_args()
    print(f'Creating model {args.model}')
    model = create_model(args.model, divided_depths=args.divided_depths)
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt['model'])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(f'Spliting {args.model} into {len(model.stages)} modules...')
    for i, module in enumerate(model.stages):
        save_path = os.path.join(args.save_dir, f'{args.model}_{i}.pth')
        torch.save({
            'model_name': args.model,
            'block_index': i,
            'block_state_dict':module.state_dict(),
            'divided_depths': args.divided_depths,
        }, save_path)
    print('Done')
