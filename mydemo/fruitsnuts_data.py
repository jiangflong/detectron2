from os.path import dirname, abspath
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
# 获取根目录
base_dir = dirname(dirname(abspath(__file__)))
# 修改成linux目录
base_dir = base_dir.replace('\\', '/')
dataDir = "/datasets/coco"
register_coco_instances("fruits_nuts", {}, dataDir+"/data/trainval.json", dataDir+"/data/images")