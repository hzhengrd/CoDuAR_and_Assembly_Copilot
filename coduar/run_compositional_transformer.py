# Training script for compositional dual-hand model with transformer decoder
import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model
from timm.utils import ModelEma

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models  # noqa: F401
import models.modeling_finetune_compositional_dual_transformer  # Import transformer model
import utils
from dataset.build_compositional import build_compositional_dual_hand_datasets
from engine_for_compositional_transformer import (
    train_one_epoch_compositional_transformer,
    validation_one_epoch_compositional,
    final_test_compositional,
)
from optim_factory import (
    LayerDecayValueAssigner,
    create_optimizer,
)
from utils import NativeScalerWithGradNormCount as NativeScaler


def get_args():
    parser = argparse.ArgumentParser(
        'Compositional Dual-hand VideoMAE with Transformer Decoder', add_help=False)
    
    # Basic parameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224_compositional_dual_transformer', type=str,
                        help='Model with transformer decoder for element communication')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--with_checkpoint', action='store_true', default=False)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')
    parser.add_argument('--head_drop_rate', type=float, default=0.1, metavar='PCT')

    # Transformer decoder parameters
    parser.add_argument('--decoder_layers', type=int, default=3,
                       help='Number of transformer decoder layers')
    parser.add_argument('--decoder_heads', type=int, default=8,
                       help='Number of attention heads in decoder')
    parser.add_argument('--decoder_dim', type=int, default=2048,
                       help='FFN dimension in decoder')
    parser.add_argument('--decoder_dropout', type=float, default=0.1,
                       help='Dropout in decoder')

    # Hand adapter parameters
    parser.add_argument('--use_hand_adapters', action='store_true', default=True)
    parser.add_argument('--no_hand_adapters', action='store_false', dest='use_hand_adapters')
    parser.add_argument('--use_shared_adapter', action='store_true', default=False,
                        help='Use one shared adapter for both hands instead of hand-specific adapters '
                             '(ablation study). Automatically sets --no_hand_adapters.')
    parser.add_argument('--adapter_dim', type=int, default=128)
    parser.add_argument('--no_element_self_attn', action='store_true', default=False,
                        help='Remove element self-attention from decoder layers '
                             '(ablation study). Cross-attention and FFN are kept.')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)
    
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--warmup_lr', type=float, default=1e-8, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT')
    parser.add_argument('--num_sample', type=int, default=2)
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--train_interpolation', type=str, default='bicubic')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=10)
    parser.add_argument('--test_num_crop', type=int, default=3)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT')
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Compositional dual-hand dataset parameters
    parser.add_argument('--lh_data_dir', required=True, type=str)
    parser.add_argument('--rh_data_dir', required=True, type=str)
    parser.add_argument('--lh_train_ann', required=True, type=str)
    parser.add_argument('--rh_train_ann', required=True, type=str)
    parser.add_argument('--lh_val_ann', required=True, type=str)
    parser.add_argument('--rh_val_ann', required=True, type=str)
    
    # Extra validation datasets (optional)
    parser.add_argument('--lh_extra_val_data_dir', default=None, type=str,
                       help='Optional extra validation data directory for left hand')
    parser.add_argument('--rh_extra_val_data_dir', default=None, type=str,
                       help='Optional extra validation data directory for right hand')
    parser.add_argument('--lh_extra_val_ann', default=None, type=str,
                       help='Optional extra validation annotation file for left hand')
    parser.add_argument('--rh_extra_val_ann', default=None, type=str,
                       help='Optional extra validation annotation file for right hand')
    parser.add_argument('--extra_val_name', default='extra', type=str,
                       help='Name prefix for extra validation metrics')
    
    # Compositional label class counts
    parser.add_argument('--lh_num_verbs', default=20, type=int)
    parser.add_argument('--lh_num_manip_objs', default=50, type=int)
    parser.add_argument('--lh_num_target_objs', default=50, type=int)
    parser.add_argument('--lh_num_tools', default=20, type=int)
    parser.add_argument('--rh_num_verbs', default=20, type=int)
    parser.add_argument('--rh_num_manip_objs', default=50, type=int)
    parser.add_argument('--rh_num_target_objs', default=50, type=int)
    parser.add_argument('--rh_num_tools', default=20, type=int)
    
    # Standard dataset params
    parser.add_argument('--data_set', default='HAVID', type=str)
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--sparse_sample', default=False, action='store_true')
    parser.add_argument('--fname_tmpl', default='img_{:05}.jpg', type=str)
    parser.add_argument('--start_idx', default=1, type=int)

    # Output
    parser.add_argument('--output_dir', default='', help='path where to save')
    parser.add_argument('--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser.parse_args()


def main(args):
    utils.init_distributed_mode(args)
    
    print(args)
    print("\n" + "="*60)
    print("COMPOSITIONAL DUAL-HAND WITH TRANSFORMER DECODER")
    print("="*60)
    print(f"Decoder Layers: {args.decoder_layers}")
    print(f"Decoder Heads: {args.decoder_heads}")
    print(f"Hand Adapters: {'ENABLED' if args.use_hand_adapters else 'DISABLED'}")
    print("="*60 + "\n")
    
    device = torch.device(args.device)
    
    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # Build compositional dual-hand datasets
    dataset_train, _ = build_compositional_dual_hand_datasets(is_train=True, test_mode=False, args=args)
    
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val, _ = build_compositional_dual_hand_datasets(is_train=False, test_mode=False, args=args)
    
    dataset_test, _ = build_compositional_dual_hand_datasets(is_train=False, test_mode=True, args=args)
    
    # Build extra validation datasets if specified
    dataset_extra_val = None
    if (args.lh_extra_val_data_dir and args.rh_extra_val_data_dir and 
        args.lh_extra_val_ann and args.rh_extra_val_ann):
        print(f"Building extra validation dataset: {args.extra_val_name}")
        # Temporarily replace val annotations and data dirs
        orig_lh_val_ann = args.lh_val_ann
        orig_rh_val_ann = args.rh_val_ann
        orig_lh_data_dir = args.lh_data_dir
        orig_rh_data_dir = args.rh_data_dir
        
        args.lh_val_ann = args.lh_extra_val_ann
        args.rh_val_ann = args.rh_extra_val_ann
        args.lh_data_dir = args.lh_extra_val_data_dir
        args.rh_data_dir = args.rh_extra_val_data_dir
        
        dataset_extra_val, _ = build_compositional_dual_hand_datasets(is_train=False, test_mode=True, args=args)
        
        # Restore original values
        args.lh_val_ann = orig_lh_val_ann
        args.rh_val_ann = orig_rh_val_ann
        args.lh_data_dir = orig_lh_data_dir
        args.rh_data_dir = orig_rh_data_dir

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))
    
    if args.dist_eval:
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False) if dataset_val else None
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_extra_val = torch.utils.data.DistributedSampler(
            dataset_extra_val, num_replicas=num_tasks, rank=global_rank, shuffle=False) if dataset_extra_val else None
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val else None
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        sampler_extra_val = torch.utils.data.SequentialSampler(dataset_extra_val) if dataset_extra_val else None

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # Data loaders
    # Use custom collate function for compositional dual-hand data
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True,
        collate_fn=utils.compositional_dual_hand_collate
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True,
            collate_fn=utils.compositional_dual_hand_collate
        )
    else:
        data_loader_val = None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True,
        collate_fn=utils.compositional_dual_hand_collate
    )
    
    if dataset_extra_val is not None:
        data_loader_extra_val = torch.utils.data.DataLoader(
            dataset_extra_val,
            sampler=sampler_extra_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True,
            collate_fn=utils.compositional_dual_hand_collate
        )
    else:
        data_loader_extra_val = None

    # Create model with transformer decoder
    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False,
        lh_num_verbs=args.lh_num_verbs,
        lh_num_manip_objs=args.lh_num_manip_objs,
        lh_num_target_objs=args.lh_num_target_objs,
        lh_num_tools=args.lh_num_tools,
        rh_num_verbs=args.rh_num_verbs,
        rh_num_manip_objs=args.rh_num_manip_objs,
        rh_num_target_objs=args.rh_num_target_objs,
        rh_num_tools=args.rh_num_tools,
        use_hand_adapters=args.use_hand_adapters and not args.use_shared_adapter,
        use_shared_adapter=args.use_shared_adapter,
        adapter_dim=args.adapter_dim,
        use_element_self_attn=not args.no_element_self_attn,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_dim=args.decoder_dim,
        decoder_dropout=args.decoder_dropout,
        head_dropout=args.head_drop_rate,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        with_cp=args.with_checkpoint,
    )

    # Load pretrained weights
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        
        print("Load pretrained ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        
        # Remove old format keys
        for old_key in list(checkpoint_model.keys()):
            if old_key.startswith('_orig_mod.'):
                new_key = old_key[10:]
                checkpoint_model[new_key] = checkpoint_model.pop(old_key)
        
        # Remove incompatible head and decoder weights
        keys_to_remove = []
        for k in list(checkpoint_model.keys()):
            if any(x in k for x in ['head.', 'decoder.', 'adapter.']):
                keys_to_remove.append(k)
        
        for k in keys_to_remove:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
        
        # Handle backbone prefix
        all_keys = list(checkpoint_model.keys())
        new_dict = {}
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict
        
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=''
        )

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters
    encoder_params = sum(p.numel() for n, p in model.named_parameters() 
                        if 'head' not in n and 'adapter' not in n and 'decoder' not in n and p.requires_grad)
    head_params = sum(p.numel() for n, p in model.named_parameters() if 'head' in n and p.requires_grad)
    adapter_params = sum(p.numel() for n, p in model.named_parameters() if 'adapter' in n and p.requires_grad)
    decoder_params = sum(p.numel() for n, p in model.named_parameters() if 'decoder' in n and p.requires_grad)
    
    print("Model = %s" % str(model_without_ddp))
    print('Total params: {:,}'.format(n_parameters))
    print('  Encoder params: {:,}'.format(encoder_params))
    print('  Element head params: {:,}'.format(head_params))
    print('  Hand adapter params: {:,}'.format(adapter_params))
    print('  Transformer decoder params: {:,}'.format(decoder_params))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    
    total_batch_size = args.batch_size * args.update_freq * num_tasks
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay**(num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    skip_weight_decay_list = model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, 'no_weight_decay') else []

    optimizer = create_optimizer(
        args,
        model_without_ddp,
        skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None
    )
    loss_scaler = NativeScaler()

    # Learning rate schedule
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)

    # Auto resume
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    # Validation only mode
    if args.validation:
        test_stats = validation_one_epoch_compositional(data_loader_val, model, device)
        exit(0)

    # Evaluation only mode
    if args.eval:
        file_prefix = os.path.join(args.output_dir, str(global_rank))
        test_stats = final_test_compositional(data_loader_test, model, device, file_prefix)
        exit(0)

    # Training loop
    print(f"Start training with transformer decoder for {args.epochs} epochs")
    start_time = time.time()
    best_accuracy = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        
        train_stats = train_one_epoch_compositional_transformer(
            model, criterion, data_loader_train, optimizer, device, epoch,
            loss_scaler, args.clip_grad, model_ema, None,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
        )
        
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or (epoch + 1) == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                    model_ema=model_ema)
        
        if data_loader_val is not None:
            test_stats = validation_one_epoch_compositional(data_loader_val, model, device)
            
            # Use whole action accuracy for best model selection
            avg_whole_action_accuracy = (
                test_stats.get('lh_whole_action_top1', 0) + 
                test_stats.get('rh_whole_action_top1', 0)
            ) / 2.0
            
            if avg_whole_action_accuracy > best_accuracy and args.output_dir and args.save_ckpt:
                best_accuracy = avg_whole_action_accuracy
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch="best",
                    model_ema=model_ema)
                
                # Save validation metrics
                best_metrics_file = os.path.join(args.output_dir, 'best_model_val_metrics.json')
                with open(best_metrics_file, 'w') as f:
                    json.dump(test_stats, f, indent=2)
                print(f'Best model metrics saved to {best_metrics_file}')
            
            print(f'Best whole action accuracy: {best_accuracy:.2f}%')
            
            if log_writer is not None:
                # Log all metrics
                for hand in ['lh', 'rh']:
                    for element in ['verb', 'manip_obj', 'target_obj', 'tool']:
                        log_writer.update(**{f'val_{hand}_{element}_top1': test_stats[f'{hand}_{element}_top1']}, 
                                        head="perf", step=epoch)
                        log_writer.update(**{f'val_{hand}_{element}_top5': test_stats[f'{hand}_{element}_top5']}, 
                                        head="perf", step=epoch)
                
                log_writer.update(val_lh_whole_action_top1=test_stats['lh_whole_action_top1'], head="perf", step=epoch)
                log_writer.update(val_lh_whole_action_top5=test_stats['lh_whole_action_top5'], head="perf", step=epoch)
                log_writer.update(val_rh_whole_action_top1=test_stats['rh_whole_action_top1'], head="perf", step=epoch)
                log_writer.update(val_rh_whole_action_top5=test_stats['rh_whole_action_top5'], head="perf", step=epoch)
                log_writer.update(val_loss=test_stats['loss'], head="perf", step=epoch)
        
        # Evaluate on extra validation dataset if available
        extra_val_stats = None
        if data_loader_extra_val is not None:
            extra_val_stats = validation_one_epoch_compositional(data_loader_extra_val, model, device)
            
            # Compute average whole action accuracy for extra val
            extra_val_avg_whole_action_accuracy = (
                extra_val_stats.get('lh_whole_action_top1', 0) + 
                extra_val_stats.get('rh_whole_action_top1', 0)
            ) / 2.0
            
            print(f'\n{args.extra_val_name} Validation Results:')
            print(f'  Average Whole Action Accuracy: {extra_val_avg_whole_action_accuracy:.2f}%')
            
            if log_writer is not None:
                # Log extra validation metrics with custom prefix
                for hand in ['lh', 'rh']:
                    for element in ['verb', 'manip_obj', 'target_obj', 'tool']:
                        log_writer.update(**{f'{args.extra_val_name}_{hand}_{element}_top1': extra_val_stats[f'{hand}_{element}_top1']}, 
                                         head="perf", step=epoch)
                        log_writer.update(**{f'{args.extra_val_name}_{hand}_{element}_top5': extra_val_stats[f'{hand}_{element}_top5']}, 
                                         head="perf", step=epoch)
                
                log_writer.update(**{f'{args.extra_val_name}_lh_whole_action_top1': extra_val_stats['lh_whole_action_top1']}, head="perf", step=epoch)
                log_writer.update(**{f'{args.extra_val_name}_lh_whole_action_top5': extra_val_stats['lh_whole_action_top5']}, head="perf", step=epoch)
                log_writer.update(**{f'{args.extra_val_name}_rh_whole_action_top1': extra_val_stats['rh_whole_action_top1']}, head="perf", step=epoch)
                log_writer.update(**{f'{args.extra_val_name}_rh_whole_action_top5': extra_val_stats['rh_whole_action_top5']}, head="perf", step=epoch)
                log_writer.update(**{f'{args.extra_val_name}_avg_whole_action_top1': extra_val_avg_whole_action_accuracy}, head="perf", step=epoch)
                log_writer.update(**{f'{args.extra_val_name}_loss': extra_val_stats['loss']}, head="perf", step=epoch)

        # Log stats
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        if data_loader_val is not None:
            log_stats.update({f'val_{k}': v for k, v in test_stats.items()})
        if extra_val_stats is not None:
            log_stats.update({f'{args.extra_val_name}_{k}': v for k, v in extra_val_stats.items()})
        
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Final test
    print("\n" + "="*60)
    print("Running final test...")
    print("="*60)
    file_prefix = os.path.join(args.output_dir, 'final_test')
    test_stats = final_test_compositional(data_loader_test, model, device, file_prefix)
    
    # Final test on extra validation dataset if available
    extra_val_test_stats = None
    if data_loader_extra_val is not None:
        print("\n" + "="*60)
        print(f"Running final test on {args.extra_val_name} dataset...")
        print("="*60)
        extra_file_prefix = os.path.join(args.output_dir, f'final_test_{args.extra_val_name}')
        extra_val_test_stats = final_test_compositional(data_loader_extra_val, model, device, extra_file_prefix)
    
    # Save final summary
    if utils.is_main_process():
        summary_file = os.path.join(args.output_dir, 'final_test_summary.json')
        summary_data = {
            'test_metrics': test_stats,
            'training_time': str(datetime.timedelta(seconds=int(time.time() - start_time))),
            'best_val_whole_action_acc': best_accuracy,
            'num_epochs': args.epochs,
            'model': args.model,
            'decoder_layers': args.decoder_layers,
            'decoder_heads': args.decoder_heads,
        }
        
        if extra_val_test_stats is not None:
            summary_data[f'{args.extra_val_name}_test_metrics'] = extra_val_test_stats
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f'\nResults saved to {summary_file}')
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('\n' + "="*60)
    print('Training time {}'.format(total_time_str))
    print(f'Best validation whole action accuracy: {best_accuracy:.2f}%')
    
    if extra_val_test_stats is not None:
        extra_val_avg = (
            extra_val_test_stats.get('lh_whole_action_top1', 0) + 
            extra_val_test_stats.get('rh_whole_action_top1', 0)
        ) / 2.0
        print(f'{args.extra_val_name} final test whole action accuracy: {extra_val_avg:.2f}%')
    
    print("="*60)


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)

