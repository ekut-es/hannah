import torch
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

trainer = Trainer(gpus=1, deterministic=True, logger=False)
seed_everything(1234, workers=True)
checkpoint_path = '../../trained_models/test_scope/pv_rcnn_plus_plus/checkpoints/'
checkpoint_name = 'epoch=73-step=66600.ckpt'
checkpoint = torch.load(checkpoint_path+checkpoint_name, map_location="cuda:0")
hparams = checkpoint["hyper_parameters"]
hparams['dataset']['DATA_PATH'] = '/home/sven/projects/hannah/datasets/scope/sensor_data_v3/'
hparams["_target_"] = "hannah.modules.lidar_detection.LidarDetectionModule"

vehicles = ['VEHICLE_00', 'VEHICLE_01', 'VEHICLE_02', 'VEHICLE_03']
for vehicle in vehicles:
    hparams['dataset']['SAVE_DETECTIONS'] = True
    hparams['dataset']['SAVE_DETECTION_VEHICLE'] = vehicle
    module = instantiate(hparams, _recursive_=False)
    module.setup('test')
    module.cuda()
    module.load_state_dict(checkpoint["state_dict"])
    val = trainer.validate(model=module, ckpt_path=None, verbose=True)
