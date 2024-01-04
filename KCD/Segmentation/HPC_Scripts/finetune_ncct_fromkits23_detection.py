import os
os.environ['OV_DATA_BASE'] = "/Users/mcgoug01/Library/CloudStorage/OneDrive-CRUKCambridgeInstitute/SecondYear/Segmentation/seg_data"
from KCD.Segmentation.ovseg.model.SegmentationModel import SegmentationModel
from KCD.Segmentation.ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from KCD.Segmentation.ovseg.networks.UNet import Logits
import gc
import torch
import sys


data_name = 'masked_coreg_ncct'
spacing = 2
fold = 0

pretrain_name = 'kits23_nooverlap'
# preprocessed_name = '4mm_binary'
preprocessed_name = '2mm_binary'
# preprocessed_name='4mm_binary_test'
model_name = '6,3x3x3,32_finetune_fromkits23no_detection'


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
n_fg_classes = 3
    


model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=z_to_xy_ratio,
                                                     n_fg_classes=n_fg_classes,
                                                     use_prg_trn=False)


model_params['architecture'] = 'UNet'
model_params['network']['kernel_sizes'] =6*[(3,3,3)]
model_params['network']['norm'] = 'inst'
model_params['network']['in_channels']=1
model_params['network']['filters']=32
model_params['network']['filters_max']=1024
del model_params['network']['block']
del model_params['network']['z_to_xy_ratio']
del model_params['network']['n_blocks_list']
del model_params['network']['stochdepth_rate']

lr=0.0001
model_params['data']['folders'] = ['images', 'labels']
model_params['data']['keys'] = ['image', 'label']
model_params['training']['num_epochs'] = 100
model_params['training']['opt_name'] = 'ADAM'
model_params['training']['opt_params'] = {'lr': lr,
                                            'betas': (0.95, 0.9),
                                            'eps': 1e-08}
model_params['training']['lr_params'] = {'n_warmup_epochs': 15, 'lr_max': 0.0005}
model_params['data']['trn_dl_params']['epoch_len']=250
model_params['data']['trn_dl_params']['padded_patch_size']=[2*patch_size[0]]*3
model_params['data']['val_dl_params']['padded_patch_size']=[2*patch_size[0]]*3
model_params['training']['lr_schedule'] = 'lin_ascent_log_decay'
model_params['training']['lr_exponent'] = 3
model_params['data']['trn_dl_params']['batch_size']=16
model_params['data']['val_dl_params']['epoch_len']=50

alternate_model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=z_to_xy_ratio,
                                                     n_fg_classes=1,
                                                     use_prg_trn=False)

alternate_model_params['architecture'] = 'UNet'
alternate_model_params['network']['kernel_sizes'] =6*[(3,3,3)]
alternate_model_params['network']['norm'] = 'inst'
alternate_model_params['network']['in_channels']=1
alternate_model_params['network']['filters']=32
alternate_model_params['network']['filters_max']=1024
del alternate_model_params['network']['block']
del alternate_model_params['network']['z_to_xy_ratio']
del alternate_model_params['network']['n_blocks_list']
del alternate_model_params['network']['stochdepth_rate']

lr=0.0001
alternate_model_params['data']['folders'] = ['images', 'labels']
alternate_model_params['data']['keys'] = ['image', 'label']
alternate_model_params['training']['num_epochs'] = 100
alternate_model_params['training']['opt_name'] = 'ADAM'
alternate_model_params['training']['opt_params'] = {'lr': lr,
                                            'betas': (0.95, 0.9),
                                            'eps': 1e-08}
alternate_model_params['training']['lr_params'] = {'n_warmup_epochs': 15, 'lr_max': 0.0005}
alternate_model_params['data']['trn_dl_params']['epoch_len']=250
alternate_model_params['data']['trn_dl_params']['padded_patch_size']=[2*patch_size[0]]*3
alternate_model_params['data']['val_dl_params']['padded_patch_size']=[2*patch_size[0]]*3
alternate_model_params['training']['lr_schedule'] = 'lin_ascent_log_decay'
alternate_model_params['training']['lr_exponent'] = 3
alternate_model_params['data']['trn_dl_params']['batch_size']=16
alternate_model_params['data']['val_dl_params']['epoch_len']=50



for vf in [0,1,2,3,4]:
    path_to_model = '{}/trained_models/{}/2mm_binary/6,3x3x3,32/fold_4/network_weights'.format(os.environ['OV_DATA_BASE'],
                                                                                         pretrain_name)
    #
    model = SegmentationModel(val_fold=vf,
                                data_name=data_name,
                                preprocessed_name=preprocessed_name,
                                model_name=model_name,
                                model_parameters=model_params)
    #
    model.network.load_state_dict(torch.load(path_to_model,map_location=dev))
    #
    for i,log_layer in enumerate(model.network.all_logits):
        model.network.all_logits[i] = Logits(log_layer.logits.in_channels,
                                              2,
                                              False)

    alternate_model = SegmentationModel(val_fold=vf,
                                data_name=data_name,
                                preprocessed_name=preprocessed_name,
                                model_name=model_name,
                                model_parameters=alternate_model_params)

    alternate_model.network.load_state_dict(model.network.state_dict())

    alternate_model.training.train()
    alternate_model.eval_validation_set(continuous=True)
