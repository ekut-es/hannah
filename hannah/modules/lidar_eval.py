import torch
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

trainer = Trainer(gpus=1, deterministic=True, logger=False)
seed_everything(1234, workers=True)
checkpoint_path = '../../trained_models/test_01/voxel_rcnn/checkpoints/'
checkpoint_name = 'epoch=78-step=81370.ckpt'
checkpoint = torch.load(checkpoint_path+checkpoint_name, map_location="cuda:0")
hparams = checkpoint["hyper_parameters"]

splits = [['test_clear_day']]

# p = hparams['model']['MODEL']['ROI_HEAD']['ROI_GRID_POOL']['POOL_LAYERS']
# for name, config in p.items():
#     config['MLPS'] = [config['MLPS'][0][1:]]

hparams["_target_"] = "hannah.modules.lidar_detection.LidarDetectionModule"
results = []

for split in splits:
    hparams['dataset']['DATA_SPLIT']['test'] = split
    hparams['dataset']['INFO_PATH']['test'] = ['infos_' + s + '.pkl' for s in split]
    module = instantiate(hparams, _recursive_=False)
    module.setup('test')
    module.cuda()
    module.load_state_dict(checkpoint["state_dict"])
    val = trainer.validate(model=module, ckpt_path=None, verbose=True)
    results.append(val[0]['AP'])

# augmentation_name = 'baseline'
# Car_3D = augmentation_name
# Car_2D = augmentation_name
# Car_AOS = augmentation_name
# Car_BEV = augmentation_name
#
# for res in results:
#     Car_3D += ' & ' + str(round(res['Car_3d/easy_R40'].item(), 2)) + ' & ' + str(round(res['Car_3d/moderate_R40'].item(), 2)) + ' & ' + str(round(res['Car_3d/hard_R40'].item(), 2))
#     Car_2D += ' & ' + str(round(res['Car_image/easy_R40'].item(), 2)) + ' & ' + str(round(res['Car_image/moderate_R40'].item(), 2)) + ' & ' + str(round(res['Car_image/hard_R40'].item(), 2))
#     Car_AOS += ' & ' + str(round(res['Car_aos/easy_R40'].item(), 2)) + ' & ' + str(round(res['Car_aos/moderate_R40'].item(), 2)) + ' & ' + str(round(res['Car_aos/hard_R40'].item(), 2))
#     Car_BEV += ' & ' + str(round(res['Car_bev/easy_R40'].item(), 2)) + ' & ' + str(round(res['Car_bev/moderate_R40'].item(), 2)) + ' & ' + str(round(res['Car_bev/hard_R40'].item(), 2))
#
# print('----------------results for datasets ------------')
# print(splits)
# print('-------------------------------------------------')
# print('---------------------- Car 3D -------------------')
# print(Car_3D)
# print('-------------------------------------------------')
# print('---------------------- Car 2D -------------------')
# print(Car_2D)
# print('-------------------------------------------------')
# print('---------------------- Car BEV ------------------')
# print(Car_BEV)
# print('-------------------------------------------------')
# print('---------------------- Car AOS ------------------')
# print(Car_AOS)
# print('-------------------------------------------------')
