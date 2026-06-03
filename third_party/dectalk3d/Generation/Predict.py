import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import numpy as np
from tqdm import tqdm
from FLAME.FLAME import FLAME
from Generation.FaceGeneration import FaceGenerationModel
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
    model = FaceGenerationModel(
        config['predict']['vqvae_dir'],
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
    flame_model = FLAME(Config(
        config['predict']['shape'],
        config['predict']['expression'],
        config['flame_model'],
        config['static_landmark_embedding'],
        config['dynamic_landmark_embedding'],
    )).to(device)
    dataloader = get_dataloader(config['test_file_path'], batch_size=1)
    model.load_state_dict(torch.load(config['predict']['generation_dir'], map_location="cuda:0")['model_state_dict'])
    seq_template = torch.tensor(np.load(config['template'])).to(device).unsqueeze(0)
    # 设置保存路径
    save_path = config['predict']['save_path']
    os.makedirs(save_path, exist_ok=True)

    model.eval()
    vertices_gt_all, vertices_output_all, total_mee, total_ce, diversity_scores = [], [], [], [], []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for i, (video_token, person_one_hot, _, text, audio, shape, exp, jaw, mask) in pbar:
            # 数据预处理
            person_one_hot, audio, shape, exp, jaw, n = (person_one_hot.to(device), audio.to(device),
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
                exp_outputs, jaw_outputs = model.predict(person_one_hot, text, audio)
                pose_outputs_params = torch.cat((pose_params[:, :3], jaw_outputs.squeeze(0)), dim=1)
                vertices_output, _ = flame_model(shape.squeeze(0), exp_outputs.squeeze(0), pose_outputs_params)
                vertices_output = vertices_output[:n, :, :].cpu().numpy()
                vertices_output_current.append(vertices_output)
            # 保存最后一次生成结果，用于计算MVE，LVE, FDD
            vertices_output_all.extend(list(vertices_output))
            # 保存最后一次生成结果到文件
            test_file_path = os.path.join(save_path, f"{video_token[0]}.npy")
            np.save(test_file_path, vertices_output)

            # 计算 MEE
            vertices_output_mean = np.mean(np.array(vertices_output_current), axis=0)  # 计算多次生成结果的平均值
            mee = np.mean(lve_compute(vertices_gt, vertices_output_mean, mouth_map))
            total_mee.append(mee)

            # 计算 CE
            ce = np.min([np.mean(lve_compute(vertices_gt, gen, mouth_map)) for gen in vertices_output_current])
            total_ce.append(ce)

            # 计算 Diversity
            np.random.shuffle(vertices_output_current)
            subset1 = vertices_output_current[:5]
            subset2 = vertices_output_current[5:]
            diversity_value = np.mean([np.linalg.norm(sample1 - sample2, axis=2).mean()
                                       for sample1, sample2 in zip(subset1, subset2)])
            diversity_scores.append(diversity_value)

        # 计算MVE，LVE, FDD
        mve = mve_compute(vertices_gt_all, vertices_output_all)
        lve = lve_compute(vertices_gt_all, vertices_output_all, mouth_map)
        fdd = fdd_compute(vertices_gt_all, vertices_output_all, upper_map, seq_template.cpu().numpy())

        # 输出最终结果
        print(f"MVE: {np.mean(mve):.4e}")
        print(f"LVE: {np.mean(lve):.4e}")
        print(f"FDD: {np.mean(fdd):.4e}")
        print(f"MEE: {np.mean(total_mee):.4e}")
        print(f"CE: {np.mean(total_ce):.4e}")
        print(f"Diversity: {np.mean(diversity_scores):.4e}")


# 执行预测
if __name__ == "__main__":
    predict_model('config.yaml')
