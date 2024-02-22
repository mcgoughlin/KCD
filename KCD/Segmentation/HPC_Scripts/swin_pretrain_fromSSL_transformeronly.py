import os
os.environ['OV_DATA_BASE'] = "/bask/projects/p/phwq4930-renal-canc/data/seg_data"
# os.environ['OV_DATA_BASE'] = "/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data"
from KCD.Segmentation.ovseg.model.SegmentationModel import SegmentationModel
from KCD.Segmentation.ovseg.model.model_parameters_segmentation import get_model_params_3d_swinunetr
import gc
import torch
import sys

SSL_model_name = 'maskless_multirecloss_1gpu_700_10_1_5e-05'
data_name = 'kits23_nooverlap'
spacing = 4
fold = int(sys.argv[1])
# fold = 0

# preprocessed_name = '4mm_binary'
preprocessed_name = '4mm_binary'
# preprocessed_name='4mm_binary_test'
model_name = 'swinpretrain_fromSSL-{}_singlelayerdecoder_test'.format(SSL_model_name)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
vfs = [fold]

patch_size = [64,64,64]
#patch dimension must be divisible by respective (((kernel_dimension+1)//2)^depth)/2
#Patch size dictates input size to CNN: input dim (metres) = patch_size*target_spacing/1000
#finally, depth and conv kernel size dictate attentive area - importantly different to input size:
# attentive_area (in each dimension, metres) = input size / bottom encoder spatial dim
#                                           = ((((kernel_dimension+1)//2)^depth)/2)*target_spacing/1000
z_to_xy_ratio = 1
larger_res_encoder = True
n_fg_classes = 1
    


model_params = get_model_params_3d_swinunetr(patch_size,
                                                     z_to_xy_ratio=z_to_xy_ratio,
                                                     n_fg_classes=n_fg_classes,
                                                     use_prg_trn=False)

lr=0.0001

model_params['data']['folders'] = ['images', 'labels']
model_params['data']['keys'] = ['image', 'label']
model_params['training']['num_epochs'] = 100 #100
model_params['training']['opt_name'] = 'ADAM'
model_params['training']['opt_params'] = {'lr': lr,
                                            'betas': (0.95, 0.9),
                                            'eps': 1e-08}
model_params['training']['lr_params'] = {'n_warmup_epochs': 15, 'lr_max': 0.0005} #0.0005
model_params['data']['trn_dl_params']['epoch_len']=250 #250
model_params['data']['trn_dl_params']['padded_patch_size']=[2*patch_size[0]]*3
model_params['data']['val_dl_params']['padded_patch_size']=[2*patch_size[0]]*3
model_params['training']['lr_schedule'] = 'lin_ascent_log_decay'
model_params['training']['lr_exponent'] = 3
model_params['data']['trn_dl_params']['batch_size']=16
model_params['data']['val_dl_params']['epoch_len']=50
model_params['data']['trn_dl_params']['num_workers']=0
model_params['data']['val_dl_params']['num_workers']=0


for vf in vfs:
    model = SegmentationModel(val_fold=vf,
                                data_name=data_name,
                                preprocessed_name=preprocessed_name, 
                                model_name=model_name,
                                model_parameters=model_params)
    #
    pretrained_model = torch.load('/bask/projects/p/phwq4930-renal-canc/data/smit_unet/{}/pre_train_model.pt'.format(SSL_model_name), map_location=dev)
    # delete any keys in pretrained_model['state_dict'] that contain 'mask_token'
    #loop through keys in pretrained_model['state_dict'], see if key exists in model.network.state_dict(), if so, load it

    for key in pretrained_model['state_dict'].keys():
        if (key in model.network.state_dict().keys()) and ('transformer' in key):
            print(key)
            model.network.state_dict()[key].copy_(pretrained_model['state_dict'][key])
    model.training.train()
    model.eval_validation_set()
