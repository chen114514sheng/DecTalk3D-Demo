import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import numpy as np
from tqdm import tqdm
from FLAME.FLAME import FLAME
from VQVAE2 import VQVAE2
from sklearn.preprocessing import MinMaxScaler
from DataProcess.Dataload import get_dataloader
from Utils import Config, mve_compute, lve_compute, fdd_compute


def predict_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # 加载模板和嘴部/上半部分的顶点映射
    with open(config['lve']) as f:
        mouth_map = [int(i) for i in f.read().split(",")]
    with open(config['fdd']) as f:
        upper_map = [int(i) for i in f.read().split(",")]
    # 加载模型和数据集
    device = torch.device(f"cuda:{config['predict']['gpu']}" if torch.cuda.is_available() else "cpu")
    model = VQVAE2(
        config['stage1']['embed_dim'],
        config['stage1']['num_heads'],
        config['stage1']['num_layers_top'],
        config['stage1']['num_layers_bottom'],
        config['stage1']['num_layers_decoder'],
        config['stage1']['num_embeddings_top'],
        config['stage1']['num_embeddings_bottom']
    ).to(device)
    flame_model = FLAME(Config(
        config['predict']['shape'],
        config['predict']['expression'],
        config['flame_model'],
        config['static_landmark_embedding'],
        config['dynamic_landmark_embedding'],
    )).to(device)
    dataloader = get_dataloader(config['test_file_path'], batch_size=1)
    model.load_state_dict(torch.load(config['predict']['vqvae_dir'], map_location="cuda:0")['model_state_dict'])
    seq_template = torch.tensor(np.load(config['template'])).to(device).unsqueeze(0)

    model.eval()
    vertices_gt_all, vertices_output_all, total_mee, total_ce, diversity_scores = [], [], [], [], []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for i, (_, person_one_hot, _, text, _, shape, exp, jaw, mask) in pbar:
            # 数据预处理
            person_one_hot, shape, exp, jaw, n = (person_one_hot.to(device),
                                                  shape[:, :, :300].to(device), exp[:, :, :100].to(device), jaw.to(device),
                                                  int(mask.to(device).sum()))

            # 原始顶点，用于评估基准
            pose_params = torch.cat((torch.zeros((256, 3)).to(device), jaw.squeeze(0)), dim=1)
            vertices_gt, _ = flame_model(shape.squeeze(0), exp.squeeze(0), pose_params)
            vertices_gt = vertices_gt[:n, :, :].cpu().numpy()
            # 保存原始顶点，用于计算MVE，LVE
            vertices_gt_all.extend(list(vertices_gt))

            # 多次生成的采样，用于计算MEE, CE, Diversity
            vertices_output_current = []
            for sample_idx in range(10):  # 设定10次生成
                exp_outputs, jaw_outputs, _, _ = model(person_one_hot, text, exp, jaw)
                pose_outputs_params = torch.cat((pose_params[:, :3], jaw_outputs.squeeze(0)), dim=1)
                vertices_output, _ = flame_model(shape.squeeze(0), exp_outputs.squeeze(0), pose_outputs_params)
                vertices_output = vertices_output[:n, :, :].cpu().numpy()
                vertices_output_current.append(vertices_output)
            # 保存最后一次生成结果，用于计算MVE，LVE, FDD
            vertices_output_all.extend(list(vertices_output))

        # 计算MVE，LVE, FDD
        mve = mve_compute(vertices_gt_all, vertices_output_all)
        lve = lve_compute(vertices_gt_all, vertices_output_all, mouth_map)
        fdd = fdd_compute(vertices_gt_all, vertices_output_all, upper_map, seq_template.cpu().numpy())

        # 输出最终结果
        print(f"MVE: {np.mean(mve):.4e}")
        print(f"LVE: {np.mean(lve):.4e}")
        print(f"FDD: {np.mean(fdd):.4e}")


# 执行预测
if __name__ == "__main__":
    predict_model('config.yaml')
