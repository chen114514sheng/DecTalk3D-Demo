import torch
import torch.nn as nn
import torch.optim as optim
from FaceGeneration import FaceGenerationModel
from DataProcess.Dataload import get_dataloader
from Utils import ScheduledOptim
import os
import yaml
import json
import numpy as np
from tqdm import tqdm


def delete_later_checkpoints(checkpoint_dir):
    """删除 checkpoint_dir 中的所有文件"""
    for filename in os.listdir(checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def train_model(train_loader, val_loader, model, criterion, device, optimizer, embed_dim):
    """训练模型"""
    model.train()
    total_loss_train = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (_, person_one_hot, _, text, audio, _, exp, jaw, mask) in pbar:
        person_one_hot, audio, exp, jaw, mask = (person_one_hot.to(device), audio.to(device),
                                                 exp[:, :, :100].to(device), jaw.to(device), mask.to(device))
        z_top_output, z_bottom_output, z_top_pred_output, z_bottom_pred_output, exp_output, jaw_output = model(person_one_hot, text, audio, exp, jaw)
        mask_bottom, mask_top = mask[:, ::8], mask[:, ::16]
        loss_top = (criterion(z_top_pred_output, z_top_output) * mask_top.unsqueeze(2)).sum() / (embed_dim * mask_top.sum())
        loss_bottom = (criterion(z_bottom_pred_output, z_bottom_output) * mask_bottom.unsqueeze(2)).sum() / (embed_dim * mask_bottom.sum())
        loss_exp = (criterion(exp_output, exp) * mask.unsqueeze(2)).sum() / (100 * mask.sum())
        loss_jaw = (criterion(jaw_output, jaw) * mask.unsqueeze(2)).sum() / (3 * mask.sum())
        loss = 0.2 * loss_top + 0.1 * loss_bottom + 6 * loss_exp + 8 * loss_jaw
        #loss = 0.1 * loss_bottom + 6 * loss_exp + 8 * loss_jaw

        optimizer.zero_grad()
        loss.backward()
        optimizer.step_and_update_lr()

        total_loss_train.append(loss.item())
        pbar.set_description(f"Train Loss: {np.mean(total_loss_train):.4f}")

    """测试模型"""
    model.eval()
    total_loss_val = []
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    with torch.no_grad():
        for i, (_, person_one_hot, _, text, audio, _, exp, jaw, mask) in pbar:
            person_one_hot, audio, exp, jaw, mask = (person_one_hot.to(device), audio.to(device),
                                                     exp[:, :, :100].to(device), jaw.to(device), mask.to(device))
            z_top_output, z_bottom_output, z_top_pred_output, z_bottom_pred_output, exp_output, jaw_output = model(person_one_hot, text, audio, exp, jaw)
            mask_bottom, mask_top = mask[:, ::8], mask[:, ::16]
            loss_top = (criterion(z_top_pred_output, z_top_output) * mask_top.unsqueeze(2)).sum() / (embed_dim * mask_top.sum())
            loss_bottom = (criterion(z_bottom_pred_output, z_bottom_output) * mask_bottom.unsqueeze(2)).sum() / (embed_dim * mask_bottom.sum())
            loss_exp = (criterion(exp_output, exp) * mask.unsqueeze(2)).sum() / (100 * mask.sum())
            loss_jaw = (criterion(jaw_output, jaw) * mask.unsqueeze(2)).sum() / (3 * mask.sum())
            loss = 0.2 * loss_top + 0.1 * loss_bottom + 6 * loss_exp + 8 * loss_jaw
            #loss = 0.1 * loss_bottom + 6 * loss_exp + 8 * loss_jaw

            total_loss_val.append(loss.item())
            pbar.set_description(f"Val Loss: {np.mean(total_loss_val):.4f}")
    return np.mean(total_loss_train), np.mean(total_loss_val)


def main(config_path):
    """主函数"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(f"cuda:{config['stage2']['gpu']}" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = FaceGenerationModel(
        config['stage2']['vqvae_dir'],
        config['stage1']['embed_dim'],
        config['stage1']['num_heads'],
        config['stage1']['num_layers_top'],
        config['stage1']['num_layers_bottom'],
        config['stage1']['num_layers_decoder'],
        config['stage1']['num_embeddings_top'],
        config['stage1']['num_embeddings_bottom'],
        config['stage2']['num_heads'],
        config['stage2']['num_layers_top'],
        config['stage2']['num_layers_bottom']
    ).to(device)

    # 加载训练数据和测试数据
    train_dataloader = get_dataloader(config['train_file_path'], batch_size=config['stage2']['batch_size'])
    val_dataloader = get_dataloader(config['val_file_path'], batch_size=config['stage2']['batch_size'])

    # 创建保存目录，删除checkpoint_dir 中的权重和记录
    checkpoint_dir = config['stage2']['checkpoint_dir']
    record_file = os.path.join(checkpoint_dir, 'loss_record.json')
    os.makedirs(checkpoint_dir, exist_ok=True)
    delete_later_checkpoints(checkpoint_dir)
    records = {
        "best_epoch_train": 0,
        "best_loss_train": float('inf'),
        "best_epoch_val": 0,
        "best_loss_val": float('inf'),
        "train_losses": [],
        "val_losses": []
    }

    # 设置损失函数和优化器
    criterion = nn.MSELoss(reduction='none')  # 'none' 表示返回每个位置的损失
    optimizer = ScheduledOptim(optim.Adam(filter(lambda p: p.requires_grad, model.parameters())),
                               config['stage2']['learning_rate'],
                               config['stage1']['embed_dim'],
                               config['stage2']['warmup_steps'])

    # 训练循环
    epochs = config['stage2']['epochs']
    best_loss_train = records["best_loss_train"]
    best_loss_val = records["best_loss_val"]
    for epoch in range(epochs):
        train_loss, val_loss = train_model(train_dataloader, val_dataloader, model, criterion, device, optimizer,
                                           config['stage1']['embed_dim'])
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 记录并更新最佳模型
        records["train_losses"].append(train_loss)
        records["val_losses"].append(val_loss)

        if train_loss < best_loss_train:
            best_loss_train = train_loss
            records["best_epoch_train"] = epoch + 1
            records["best_loss_train"] = best_loss_train

            # 保存模型
            torch.save({
                'model_state_dict': model.state_dict(),
            }, os.path.join(checkpoint_dir, 'model_train.pth'))

        if val_loss < best_loss_val:
            best_loss_val = val_loss
            records["best_epoch_val"] = epoch + 1
            records["best_loss_val"] = best_loss_val

            # 保存模型
            torch.save({
                'model_state_dict': model.state_dict(),
            }, os.path.join(checkpoint_dir, 'model_val.pth'))

        # 保存记录文件
        with open(record_file, 'w') as f:
            json.dump(records, f, indent=4)


if __name__ == '__main__':
    main('config.yaml')
