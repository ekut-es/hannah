import torch
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
import glob

trainer = Trainer(gpus=1, deterministic=True, logger=False)
seed_everything(1234, workers=True)
# study = 'coop4_bfe_box_sampling'
study = 'coop4_bfe_box_sampling_point_deco'
# study = 'coop4_box_input_pfe'
# study = 'coop4_point_decoration'
# study = 'coop4_baseline_w_proposals_wo_augment_acc_grad_btch'
checkpoint_path = '../../trained_models/'+study+'/pv_rcnn_plus_plus/checkpoints/'
checkpoint_name = 'epoch=*.ckpt'
checkpoint_file = glob.glob(checkpoint_path+checkpoint_name)[0]
checkpoint = torch.load(checkpoint_file, map_location="cuda:0")
hparams = checkpoint["hyper_parameters"]
hparams["_target_"] = "hannah.modules.lidar_detection.LidarDetectionModule"

hparams['model'].MODEL['COOP'] = {}
hparams['model'].MODEL['COOP']['USE_COOP_PROPOSALS_IN_PFE'] = True
hparams['model'].MODEL['COOP']['USE_COOP_PROPOSALS_IN_ROI_HEAD'] = False

hparams['dataset']['LATE_FUSION'] = True

module = instantiate(hparams, _recursive_=False)
module.setup('test')
module.cuda()
module.load_state_dict(checkpoint["state_dict"])
val = trainer.validate(model=module, ckpt_path=None, verbose=True)
print(str(round(val[0]['Car_3D_AP'], 2))+' '+str(round(val[0]['Car_AP_STR'], 2)))


