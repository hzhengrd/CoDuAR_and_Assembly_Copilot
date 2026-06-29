# Training engine for compositional model with transformer decoder
# Same as basic engine but for transformer decoder model
import math
import sys
from typing import Iterable, Optional
import torch
import numpy as np
import json

from timm.data import Mixup
from timm.utils import ModelEma

import utils


def train_compositional_batch_transformer(model, samples_dict, targets_dict, criterion):
    """Train batch for compositional transformer model"""
    # Model expects dict input for dual mode
    outputs = model(samples_dict)
    
    # Compute individual losses (same as basic model)
    lh_verb_loss = criterion(outputs['lh_verb'], targets_dict['lh_verb'])
    lh_manip_obj_loss = criterion(outputs['lh_manip_obj'], targets_dict['lh_manip_obj'])
    lh_target_obj_loss = criterion(outputs['lh_target_obj'], targets_dict['lh_target_obj'])
    lh_tool_loss = criterion(outputs['lh_tool'], targets_dict['lh_tool'])
    
    rh_verb_loss = criterion(outputs['rh_verb'], targets_dict['rh_verb'])
    rh_manip_obj_loss = criterion(outputs['rh_manip_obj'], targets_dict['rh_manip_obj'])
    rh_target_obj_loss = criterion(outputs['rh_target_obj'], targets_dict['rh_target_obj'])
    rh_tool_loss = criterion(outputs['rh_tool'], targets_dict['rh_tool'])
    
    # Average element losses
    total_loss = (lh_verb_loss + lh_manip_obj_loss + lh_target_obj_loss + lh_tool_loss +
                 rh_verb_loss + rh_manip_obj_loss + rh_target_obj_loss + rh_tool_loss) / 8.0
    
    loss_dict = {
        'lh_verb_loss': lh_verb_loss.item(),
        'lh_manip_obj_loss': lh_manip_obj_loss.item(),
        'lh_target_obj_loss': lh_target_obj_loss.item(),
        'lh_tool_loss': lh_tool_loss.item(),
        'rh_verb_loss': rh_verb_loss.item(),
        'rh_manip_obj_loss': rh_manip_obj_loss.item(),
        'rh_target_obj_loss': rh_target_obj_loss.item(),
        'rh_tool_loss': rh_tool_loss.item(),
    }
    
    return total_loss, outputs, loss_dict


def train_one_epoch_compositional_transformer(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step

        # Update LR & WD
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # Handle batch data format
        if isinstance(batch_data, dict):
            lh_frames = batch_data['lh_frames'].to(device, non_blocking=True)
            rh_frames = batch_data['rh_frames'].to(device, non_blocking=True)
            lh_verb = batch_data['lh_verb'].to(device, non_blocking=True)
            lh_manip_obj = batch_data['lh_manip_obj'].to(device, non_blocking=True)
            lh_target_obj = batch_data['lh_target_obj'].to(device, non_blocking=True)
            lh_tool = batch_data['lh_tool'].to(device, non_blocking=True)
            rh_verb = batch_data['rh_verb'].to(device, non_blocking=True)
            rh_manip_obj = batch_data['rh_manip_obj'].to(device, non_blocking=True)
            rh_target_obj = batch_data['rh_target_obj'].to(device, non_blocking=True)
            rh_tool = batch_data['rh_tool'].to(device, non_blocking=True)

        samples_dict = {'lh_frames': lh_frames, 'rh_frames': rh_frames}
        targets_dict = {
            'lh_verb': lh_verb,
            'lh_manip_obj': lh_manip_obj,
            'lh_target_obj': lh_target_obj,
            'lh_tool': lh_tool,
            'rh_verb': rh_verb,
            'rh_manip_obj': rh_manip_obj,
            'rh_target_obj': rh_target_obj,
            'rh_tool': rh_tool,
        }

        if loss_scaler is None:
            samples_dict['lh_frames'] = samples_dict['lh_frames'].half()
            samples_dict['rh_frames'] = samples_dict['rh_frames'].half()
            loss, outputs, loss_dict = train_compositional_batch_transformer(
                model, samples_dict, targets_dict, criterion)
        else:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, outputs, loss_dict = train_compositional_batch_transformer(
                    model, samples_dict, targets_dict, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            grad_norm = model.get_global_grad_norm()
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(
                loss, optimizer, clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # Compute accuracy for all elements
        lh_verb_acc = (outputs['lh_verb'].max(-1)[-1] == lh_verb).float().mean()
        lh_manip_obj_acc = (outputs['lh_manip_obj'].max(-1)[-1] == lh_manip_obj).float().mean()
        lh_target_obj_acc = (outputs['lh_target_obj'].max(-1)[-1] == lh_target_obj).float().mean()
        lh_tool_acc = (outputs['lh_tool'].max(-1)[-1] == lh_tool).float().mean()
        rh_verb_acc = (outputs['rh_verb'].max(-1)[-1] == rh_verb).float().mean()
        rh_manip_obj_acc = (outputs['rh_manip_obj'].max(-1)[-1] == rh_manip_obj).float().mean()
        rh_target_obj_acc = (outputs['rh_target_obj'].max(-1)[-1] == rh_target_obj).float().mean()
        rh_tool_acc = (outputs['rh_tool'].max(-1)[-1] == rh_tool).float().mean()

        # Update metrics
        metric_logger.update(loss=loss_value)
        for key, val in loss_dict.items():
            metric_logger.update(**{key: val})
        
        metric_logger.update(lh_verb_acc=lh_verb_acc)
        metric_logger.update(lh_manip_obj_acc=lh_manip_obj_acc)
        metric_logger.update(lh_target_obj_acc=lh_target_obj_acc)
        metric_logger.update(lh_tool_acc=lh_tool_acc)
        metric_logger.update(rh_verb_acc=rh_verb_acc)
        metric_logger.update(rh_manip_obj_acc=rh_manip_obj_acc)
        metric_logger.update(rh_target_obj_acc=rh_target_obj_acc)
        metric_logger.update(rh_tool_acc=rh_tool_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            for key, val in loss_dict.items():
                log_writer.update(**{key: val}, head="loss")
            log_writer.update(lh_verb_acc=lh_verb_acc, head="loss")
            log_writer.update(lh_manip_obj_acc=lh_manip_obj_acc, head="loss")
            log_writer.update(lh_target_obj_acc=lh_target_obj_acc, head="loss")
            log_writer.update(lh_tool_acc=lh_tool_acc, head="loss")
            log_writer.update(rh_verb_acc=rh_verb_acc, head="loss")
            log_writer.update(rh_manip_obj_acc=rh_manip_obj_acc, head="loss")
            log_writer.update(rh_target_obj_acc=rh_target_obj_acc, head="loss")
            log_writer.update(rh_tool_acc=rh_tool_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# Import validation and test functions (same as other models)
from engine_for_compositional_dual_hand_finetuning_adaptive import (
    compute_top_k_accuracy,
    combine_compositional_predictions,
    elements_to_action_id,
    validation_one_epoch_compositional,
    final_test_compositional,
)

