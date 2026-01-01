# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import os
import cv2
import yaml
import torch
import wandb
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader

from network import VAENet, DepthNoise, DepthNoiseBaseline
# from dataloader import TartanAirDataLoader
from dataloader import DepthImageDataset

from skimage.metrics import structural_similarity as ssim

EPSILON = 1e-6

class LossFunction(nn.Module):
    def __init__(self, scale=1.0):
        super(LossFunction, self).__init__()        
        # reconstruction loss
        self.reco_loss = nn.HuberLoss(reduction='mean')
        # scale for the reconstruction loss
        self.scale = scale

    def forward(self, output_dict, target_x, beta=0.1):
        recon_x = output_dict['depth']
        mu, logvar = output_dict['vae']

        recon_x = recon_x.unsqueeze(1) if len(recon_x.size()) == 3 else recon_x
        target_x = target_x.unsqueeze(1) if len(target_x.size()) == 3 else target_x
        # Ensure spatial sizes match by resizing reconstruction to target size if needed
        if recon_x.size(-2) != target_x.size(-2) or recon_x.size(-1) != target_x.size(-1):
            recon_x = F.interpolate(recon_x, size=(target_x.size(-2), target_x.size(-1)), mode='bilinear', align_corners=True)
        
        # VAE loss
        kl_loss_img = torch.mean(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))
        
        # Reconstruction loss
        recon_loss = self.reco_loss(recon_x, target_x)
        
        loss_dict = {
            'recon_loss': recon_loss.item(),
            'kl_loss_img': kl_loss_img.item(),
        }
        
        # combine the losses
        final_loss = recon_loss * self.scale + beta * (kl_loss_img)
        
        return final_loss, loss_dict
    
def load_pretrained_weights(model, pretrain_path=None):
    """
    Load pre-trained weights into the model, allowing flexibility in input_size and pose_size.
    Only loads the weights that match the current model's layers.

    Args:
        model: The model into which the pre-trained weights will be loaded.
        pretrain_path (str, optional): The file path to the pre-trained weights.
    """
    try:
        state_dict = torch.load(pretrain_path, weights_only=True)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('\033[92m' + f'Loaded {len(pretrained_dict)} pre-trained weights from {pretrain_path}' + '\033[0m')
    except Exception as e:
        print(f'Failed to load pre-trained weights: {e}')
        return
    
def ssim_similarity(pred : torch.Tensor, target : torch.Tensor, max_depth : float, min_depth : float, is_log):
    pred = torch.exp(pred) - 1.0 if is_log else pred
    target = torch.exp(target) - 1.0 if is_log else target
    
    with torch.no_grad():
        rmse_loss = F.mse_loss(pred, target, reduction='mean')
        rmse_loss = torch.sqrt(rmse_loss)
        
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    # Compute SSIM for each image and average
    ssim_values = [
        ssim(pred_np[i, 0], target_np[i, 0], data_range=max_depth - min_depth)
        for i in range(target_np.shape[0])
    ]
    # average the ssim values
    return np.mean(ssim_values), rmse_loss.item()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def init_wandb(train_config, model_config, is_cluster):
    if train_config['is_wandb'] and train_config['is_train']:
        wandb.require("core")
        date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        model_type = model_config['model_type']
        run_name = train_config['run_name']
        machine = 'euler' if is_cluster else 'local'
        wandb.init(
            entity='michael-yfan24',
            project='reconstruction-vae',
            # project='reconstruction-vae-height-scan',
            name=f'{model_type}_{run_name}_{date_time_str}_{machine}',
            config={**train_config, **model_config}
        )
    return None

def train(model, data_loader, depth_noise, criterion, optimizer, scheduler, device, config, normalize_depth_input, normalize_depth_target):
    model.train()
    epochs = config['epochs']
    init_beta = config['init_beta']
    final_beta = config['final_beta']
    noise_prob = config['noise_prob']
    
    def get_beta(eopch):
        return init_beta * (final_beta / init_beta) ** (eopch / (epochs))
    
    for epoch in range(epochs):
        training_loss = 0
        beta = get_beta(epoch)
        with tqdm(enumerate(data_loader), total=len(data_loader)) as pbar:
            for iter, depth_input in pbar:                
                # Move the data to the device
                depth_input = depth_input.to(device)
                
                # Determine whether to add noise to each depth image in the batch based on noise_prob
                noise_depth = normalize_depth_input(depth_input).clone()
                noise_mask = torch.rand(noise_depth.size(0)) < noise_prob
                
                # check if the noise mask is empty
                if noise_mask.sum() > 0:
                    noise_depth[noise_mask] = depth_noise(depth_input[noise_mask])
                
                # Training loop
                optimizer.zero_grad()
                out_dict = model(noise_depth)
                loss, loss_dict = criterion(out_dict, normalize_depth_target(depth_input), beta)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                training_loss += loss.item()
                optimizer.step()
                
                pbar.set_postfix({'Training Loss': f'{training_loss / (iter + 1):.4f}', 'Beta': f'{beta:.4f}', 'Lr': f'{scheduler.get_last_lr()[0]:.6f}', 'KL Loss': f'{loss_dict["kl_loss_img"]:.4f}', 'Recon Loss': f'{loss_dict["recon_loss"]:.4f}'})
                
                if config['is_wandb']:
                    wandb.log({'Training Loss': loss.item(), 'Beta': beta, 'Average Training Loss': training_loss / (iter + 1), 'Lr': scheduler.get_last_lr()[0]})
                    for key, value in loss_dict.items():
                        wandb.log({key: value})
                        
        # Step the scheduler
        scheduler.step()
        # Save checkpoint at the end of each epoch if enabled
        if config.get('save_model', False):
            os.makedirs(os.path.join(os.getcwd(), 'model_save'), exist_ok=True)
            ckpt_path = os.path.join(
                os.getcwd(),
                'model_save',
                f"vae_{config.get('run_name', 'run')}_epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            if config.get('is_wandb', False):
                wandb.save(ckpt_path)
            
    return training_loss / (iter + 1)

def validate_and_visualize(model, data_loader, depth_noise, device, config, normalize_depth_input, normalize_depth_target, validate_fn):
    model.eval()
    with torch.no_grad():
        avg_index = 0.0
        avg_mse_loss = 0.0
        conut_imgs = 0
        val_iters = config['val_iterations']
        noise_prob = config['noise_prob']
        is_log = config['log_space']
        
        with tqdm(enumerate(data_loader), total=len(data_loader)) as pbar:
            for iter, depth_input in pbar:
                # Move the data to the device
                depth_input = depth_input.to(device)
                
                # Determine whether to add noise to each depth image in the batch based on noise_prob
                noise_depth = normalize_depth_input(depth_input).clone()
                noise_mask = torch.rand(noise_depth.size(0)) < noise_prob
                noise_depth[noise_mask] = depth_noise(depth_input[noise_mask])
                
                out_dict = model(noise_depth)
                
                depth_out = out_dict['depth']
                # Resize reconstruction to match target for fair validation and consistent saving
                if depth_out.size(-2) != depth_input.size(-2) or depth_out.size(-1) != depth_input.size(-1):
                    depth_out = F.interpolate(depth_out, size=(depth_input.size(-2), depth_input.size(-1)), mode='bilinear', align_corners=True)
                
                # compute and print the loss
                target_depth = normalize_depth_target(depth_input)
                index, rmse_loss = validate_fn(depth_out, target_depth)
                avg_index += index
                avg_mse_loss += rmse_loss
                conut_imgs += 1
                
                pbar.set_postfix({'Average Structural Similarity': f'{avg_index / conut_imgs:.4f}', 'Average RMSE Loss': f'{avg_mse_loss / conut_imgs:.4f}'})
                
                if iter == 0:
                    for b in range(depth_input.size(0)):
                        depth_input_colormap = convert_depth_to_color(noise_depth[b].cpu().numpy(), is_log)
                        depth_output_colormap = convert_depth_to_color(depth_out[b].cpu().numpy(), is_log)
                        depth_target_colormap = convert_depth_to_color(target_depth[b].cpu().numpy(), is_log)
                        # Concatenate the images horizontally
                        images = np.hstack((depth_input_colormap, depth_output_colormap, depth_target_colormap))
                        # Save the concatenated image
                        os.makedirs('images', exist_ok=True)
                        cv2.imwrite(f'images/recon_visualization_{b}.png', (images * 255).astype(np.uint8))
                        
                # break the loop if iter is greater than val_iters
                if iter >= val_iters:
                    break
    
    # Average the structural similarity index
    avg_index /= conut_imgs
    avg_mse_loss /= conut_imgs
    print("\033[93mAverage Structural Similarity: ", avg_index, "\033[0m")
    print("\033[93mAverage RMSE Loss: ", avg_mse_loss, "\033[0m")
    return None
            
def normalize_depth(depth, min_depth, max_depth, is_log, is_input=False):
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaN and Inf values
    # depth = torch.clamp(depth, min=0.0, max=max_depth)  # Clamp depth values to the specified range
    if is_input:
        depth = torch.clamp(depth, min=0.0, max=max_depth * 2.0)  # Clamp the depth values for input depth images
    else:
        depth = torch.clamp(depth, min=0.0, max=max_depth)  # Clamp the depth values for ground truth depth images
    depth = depth.unsqueeze(1) if len(depth.size()) == 3 else depth # Add channel dimension if missing
    depth = torch.log(depth + 1.0) if is_log else depth
    return depth
            
def convert_depth_to_color(depth: np.ndarray, is_log):
    depth = np.exp(depth) - 1.0 if is_log else depth
    depth = np.squeeze(depth) if len(depth.shape) == 3 else depth
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + EPSILON)
    depth = depth[:, :, np.newaxis]
    return depth
            
            
def main(config_path):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader_config = config.get('dataloader', {})
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    min_depth = dataloader_config.get('min_depth', 0.1)
    max_depth = dataloader_config.get('max_depth', 10.0)
    focal_length = dataloader_config.get('focal_length', 50.0)
    baseline = dataloader_config.get('baseline', 0.12)
    
    print("Dataloader config:", dataloader_config)
    print("Training config:", training_config)
    print("Model config:", model_config)

    # Detect whether or not is at cluster
    is_cluster = False
    if 'cluster' in os.getcwd():
        print("\033[93mRunning on cluster.\033[0m")
        is_cluster = True
        # set is_wandb to True
        training_config['is_wandb'] = True
    else:
        print("\033[93mRunning on local machine.\033[0m")        

    # data_loader = TartanAirDataLoader(dataloader_config, is_cluster)
    
    deth_dataset = DepthImageDataset(dataloader_config, is_cluster)
    data_loader = DataLoader(deth_dataset, batch_size=dataloader_config.get('batch_size', 256), shuffle=dataloader_config.get('shuffle', True), num_workers=dataloader_config.get('num_workers', 4))
    
    
    init_wandb(training_config, model_config, is_cluster)

    # Training parameters
    epochs = training_config['epochs']
    learning_rate = training_config['learning_rate']
    weight_decay = training_config['weight_decay']
    run_name = training_config['run_name']
    noise_type = training_config['noise_type']
    model_path = os.path.join(os.getcwd(), 'model_save', f'vae_{run_name}.pth')

    vae_net = VAENet(model_config['latent_dim']).to(device)
    
    optimizer = torch.optim.AdamW(vae_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = LossFunction(scale=10.0).to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Noise Model
    if noise_type == 'baseline':
        depth_noise = DepthNoiseBaseline(focal_length=focal_length, 
                                         baseline=baseline, 
                                         min_depth=min_depth, 
                                         max_depth=max_depth).to(device)
        print("\033[93mUsing baseline noise model.\033[0m")
        depth_noise = torch.jit.optimize_for_inference(torch.jit.script(depth_noise.eval()))
    elif noise_type == 'parametric':
        depth_noise = DepthNoise(focal_length=focal_length, 
                                 baseline=baseline, 
                                 min_depth=min_depth, 
                                 max_depth=max_depth).to(device)
        print("\033[93mUsing parametric noise model.\033[0m")
        depth_noise = torch.jit.optimize_for_inference(torch.jit.script(depth_noise.eval()))
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")

     # create lambda function to normalize the depth
    normalize_depth_input = lambda x: normalize_depth(x, min_depth, max_depth, is_log=training_config['log_space'], is_input=True)
    normalize_depth_target = lambda x: normalize_depth(x, min_depth, max_depth, is_log=training_config['log_space'], is_input=False)
    # create the lambda function to validate the model
    validate_fn = lambda pred, target: ssim_similarity(pred, target, max_depth, min_depth, is_log=training_config['log_space'])
    
    if training_config['load_model'] or not training_config['is_train']:
        try:
            vae_net.load_state_dict(torch.load(model_path, weights_only=True))
            print("\033[92mModel loaded successfully from path: ", model_path, "\033[0m")
        except:
            print("\033[91mModel loading failed.\033[0m")
    elif training_config['pretrain_vae']:
        # Load the pre-trained weights
        print("\033[93mLoading pre-trained VAE weights for VAE.\033[0m")
        vae_path = os.path.join(os.getcwd(), 'network/pretrain', f'vae_pretrain.pth')
        load_pretrained_weights(vae_net, pretrain_path=vae_path)
    
    if training_config['is_train']:
        train(vae_net, data_loader, depth_noise, criterion, optimizer, scheduler, device, training_config, normalize_depth_input, normalize_depth_target)
        # save the model
        if training_config.get('save_model', False):
            # get the current working directory
            torch.save(vae_net.state_dict(), model_path)
        # save the model to wandb if is_wandb is True
        if training_config['is_wandb']:
            wandb.save(model_path)
    else:
        print("\033[93mSkipping training, validating the model.\033[0m")

    validate_and_visualize(vae_net, data_loader, depth_noise, device, training_config, normalize_depth_input, normalize_depth_target, validate_fn)

    wandb.finish()
    print("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train single-frame VAE depth estimation model')
    parser.add_argument('--config', type=str, default='./config/pretrain.yaml',
                        help='Path to config file (default: ./config/pretrain.yaml)')
    args = parser.parse_args()
    main(args.config)