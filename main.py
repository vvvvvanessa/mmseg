import os.path
from mmengine import Config
from mmengine.runner import Runner
from mmseg.registry import DATASETS  # ✅ 这是官方推荐方式
from mmengine.registry import MODELS

import datasets.AsphaltDataset
import cv2
import numpy as np
import torch

#
# config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
# checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
#
# # 根据配置文件和模型文件建立模型
# model = init_model(config_file, checkpoint_file, device='cuda:0')
#
# # 在单张图像上测试并可视化
# img = 'demo/demo.JPG'  # or img = mmcv.imread(img), 这样仅需下载一次
# result = inference_model(model, img)
# # 在新的窗口可视化结果
# show_result_pyplot(model, img, result, show=True)
# # 或者将可视化结果保存到图像文件夹中
# # 您可以修改分割 map 的透明度 (0, 1].
# show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)
# # 在一段视频上测试并可视化分割结果
# # video = mmcv.VideoReader('video.mp4')
# # for frame in video:
# #    result = inference_model(model, frame)
# #    show_result_pyplot(model, frame, result, wait_time=1)

if __name__ == "__main__":
    # 加载配置文件
    config_path = 'configs/my_config.py'
    cfg = Config.fromfile(config_path)

    # 构建 Runner
    runner = Runner.from_cfg(cfg)

    # 保存原始的 train_step 方法（如果后续需要参考真实逻辑，可保留）
    orig_train_step = runner.model.train_step

    import torch


    def debug_train_step_dummy(self, data_batch, optim_wrapper):
        print("🟢 Debug train_step (dummy) called")
        print(f"data_batch keys: {data_batch.keys()}")
        inputs = data_batch['inputs']
        print(f"inputs type: {type(inputs)}")
        print(f"Number of samples: {len(inputs)}")
        if isinstance(inputs, list) and len(inputs) > 0:
            print(f"First input shape: {inputs[0].shape}")
        data_samples = data_batch['data_samples']
        print(f"data_samples type: {type(data_samples)}")
        print(f"First gt_sem_seg shape: {data_samples[0].gt_sem_seg.data.shape}")

        # 获取模型设备
        device = next(self.parameters()).device

        # 构造平展的 dummy loss 字典，每个值直接是一个 scalar tensor
        loss_seg = torch.tensor(0.0, device=device)
        loss_aux = torch.tensor(0.0, device=device)
        dummy_loss = {
            'loss_seg': loss_seg,
            'loss_aux': loss_aux,
            'loss': loss_seg + loss_aux  # 通常需要提供总 loss
        }

        print("==> Return dummy loss, skipping forward/backward")
        return dummy_loss


    # 使用 dummy loss 的 train_step 替换原始方法
    # runner.model.train_step = debug_train_step_dummy.__get__(runner.model, type(runner.model))

    # 输出模型所在设备（从模型参数中获取设备信息）
    model_device = next(runner.model.parameters()).device
    print("Model device:", model_device)

    # 开始训练
    runner.train()