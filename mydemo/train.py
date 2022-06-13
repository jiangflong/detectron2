import os
from os.path import dirname, abspath
import  fruitsnuts_data
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")


if __name__ == "__main__":
    # 获取根目录
    base_dir = dirname(dirname(abspath(__file__)))
    # 修改成linux目录
    base_dir = base_dir.replace('\\', '/')
    cfg = get_cfg()
    cfg.OUTPUT_DIR = "./output"  # 模型输出路径
    cfg.merge_from_file(
        base_dir+"/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = ("fruits_nuts",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = (2500)  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (data, fig, hazelnut)
    #cfg.MODEL.DEVICE = "cpu"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()