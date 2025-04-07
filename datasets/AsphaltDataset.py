from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class AsphaltDataset(BaseSegDataset):
  METAINFO = dict(classes = ("background", "voids"), palette = [[0, 0, 0], [255, 0, 0]])
  def __init__(self, **kwargs):
    print("start to init")
    super().__init__(ann_file = "ann_dir/train/", seg_map_suffix = ".JPG", **kwargs)
    print(f'Loaded {len(self)} images')  # 打印数据集大小
    print("ann file = " + self.ann_file)#    print(self[0])
    print("data root" + self.data_root)
    print("suffix" + self.seg_map_suffix)

