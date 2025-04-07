import os.path
from mmengine import Config
from mmengine.runner import Runner
import datasets.AsphaltDataset



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

    # config_path.get('val_dataloader')
    # config_path.get('test_dataloader')

    cfg = Config.fromfile(config_path)
    # 构建 Runner
    runner = Runner.from_cfg(cfg)

    # 开始训练
    runner.train()