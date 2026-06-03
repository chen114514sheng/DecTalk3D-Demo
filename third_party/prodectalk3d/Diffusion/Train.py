import torch
import torch.nn as nn
import torch.optim as optim
from Diffusion import FaceGenerationModel
from DataProcess.Dataload import get_dataloader
from Utils import EMA, ScheduledOptim
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


def train_model(train_loader, val_loader, model, device, optimizer, ema):
    """训练模型"""
    model.train()
    total_loss_train = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (_, person_one_hot, _, text, audio, _, exp, jaw, mask) in pbar:
        person_one_hot, audio, exp, jaw, mask = (person_one_hot.to(device), audio.to(device),
                                                 exp[:, :, :100].to(device), jaw.to(device), mask.to(device))
        loss_top, loss_bottom, loss_exp, loss_jaw = model.compute_loss(person_one_hot, text, audio, exp, jaw, mask)
        loss = 0.01 * loss_top + 0.01 * loss_bottom + 10 * loss_exp + 100 * loss_jaw

        optimizer.zero_grad()
        loss.backward()
        optimizer.step_and_update_lr()
        ema.update(model)

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
            loss_top, loss_bottom, loss_exp, loss_jaw = model.compute_loss(person_one_hot, text, audio, exp, jaw, mask)
            loss = 0.01 * loss_top + 0.01 * loss_bottom + 10 * loss_exp + 100 * loss_jaw

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
        config['stage2']['vqvae2_dir'],
        config['stage1']['embed_dim'],
        config['stage1']['num_heads'],
        config['stage1']['num_layers_style'],
        config['stage1']['num_layers_top'],
        config['stage1']['num_layers_bottom'],
        config['stage1']['num_embeddings'],
        config['stage2']['num_heads'],
        config['stage2']['num_layers_temporal'],
        config['stage2']['num_layers_semantic'],
        config['stage2']['num_layers'],
        config['stage2']['gpu']
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

    # 设置优化器
    optimizer = ScheduledOptim(optim.Adam(filter(lambda p: p.requires_grad, model.parameters())),
                               config['stage2']['learning_rate'],
                               config['stage1']['embed_dim'],
                               config['stage2']['warmup_steps'])
    ema = EMA(model, decay=0.999)
    # 训练循环
    epochs = config['stage2']['epochs']
    best_loss_train = records["best_loss_train"]
    best_loss_val = records["best_loss_val"]
    for epoch in range(epochs):
        train_loss, val_loss = train_model(train_dataloader, val_dataloader, model, device, optimizer, ema)
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
                'ema_state_dict': ema.shadow
            }, os.path.join(checkpoint_dir, 'model_train.pth'))

        if val_loss < best_loss_val:
            best_loss_val = val_loss
            records["best_epoch_val"] = epoch + 1
            records["best_loss_val"] = best_loss_val

            # 保存模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.shadow
            }, os.path.join(checkpoint_dir, 'model_val.pth'))

        # 每隔50个epoch保存一次模型权重
        if (epoch + 1) % 50 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.shadow
            }, os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth'))

        # 保存记录文件
        with open(record_file, 'w') as f:
            json.dump(records, f, indent=4)


if __name__ == '__main__':
    main('config.yaml')