from KCD.Detection.Training import train_utils as tu
import KCD.Detection.Evaluation.eval_scripts as eval_
from KCD.Detection.ModelGenerator import model_generator as mg
from sklearn.model_selection import StratifiedKFold as kfold_strat
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np
import warnings
import pandas as pd

def train_cv_slice_model(home = '/Users/mcgoug01/Downloads/Data/',dataname='coreg_ncct',
                         splits:list=[0],folds=5,params:dict=None,train_folds=[0],
                         epochs = None,learning_rate=None):
    # Suppress warnings
    warnings.filterwarnings("ignore") #makes dgl stop complaining!

    # Initialization
    dev = tu.initialize_device()

    if params==None:params = tu.init_slice3D_params_pretrain()
    else:tu.check_params(params,tu.init_slice3D_params_pretrain())
    model_type = 'PatchModel'

    if epochs != None:
        params['epochs'] = epochs

    if learning_rate != None:
        params['lr'] = learning_rate

    save_dir = tu.init_training_home(home, dataname)
    print(save_dir)

    slicedataset, test_slicedataset = tu.get_slice_data(home, dataname, params['voxel_size'], params['cancthresh_r_mm'],params['kidthresh_r_mm'],params['depth_z'],params['boundary_z'],params['dilated'],device=dev)
    cases, is_ncct = tu.get_cases(slicedataset)

    # More Initialization
    loss_fnc = nn.CrossEntropyLoss().to(dev)

    for split in splits:
        cv_results = []
        split_path = os.path.join(save_dir,'split_{}'.format(split))
        if not os.path.exists(split_path):
            os.mkdir(split_path)

        split_fp = os.path.join(split_path,'split.npy')
        five_fold_strat = kfold_strat(n_splits=folds, shuffle=True)

        if os.path.exists(split_fp): fold_split = np.load(split_fp,allow_pickle=True)
        else:
            fold_split = np.array([(fold,tr_index,ts_index) for fold,(tr_index, ts_index) in enumerate(five_fold_strat.split(cases,is_ncct))])
            np.save(os.path.join(split_path,split_fp),fold_split)

        results_csv_fold_paths = [os.path.join(split_path,'fold_{}'.format(fold),'csv','{}_{}_{}_{}_{}.csv'.format(model_type,params['model_size'],params['epochs'],params['epochs'],params['lr'])) for fold in range(folds)]

        for fold,train_index, test_index in fold_split:
            if not (fold in train_folds): continue
            fold_path = os.path.join(split_path,'fold_{}'.format(fold))
            slice_path = os.path.join(fold_path,model_type)

            if not os.path.exists(fold_path):os.mkdir(fold_path)
            if not os.path.exists(slice_path):os.mkdir(slice_path)

            model = mg.return_resnext3D(size='small',in_channels=1,out_channels=3,dev=dev)
            opt = torch.optim.Adam(model.parameters(),lr=params['lr'])

            dl,test_dl = tu.generate_dataloaders(slicedataset,test_slicedataset,cases[train_index],params['batch_size'])
            model = tu.train_model(dl,dev,params['epochs'],loss_fnc,opt,model)
            model_name = '{}_resnext_{}_{}_{}_{}'.format(model_type,params['model_size'],params['epochs'],params['epochs'],params['lr'])

            if not os.path.exists(os.path.join(slice_path,'model')):
                os.mkdir(os.path.join(slice_path,'model'))
                os.mkdir(os.path.join(slice_path,'csv'))

            torch.save(model,os.path.join(slice_path,'model',model_name))

            model.eval()
            model_res,test_df = eval_.eval_cnn(model,test_dl,dev=dev)
            cv_results.append(test_df)
            model_res.to_csv(os.path.join(slice_path,'csv',model_name+'.csv'))

        if all([os.path.exists(csv_fold_path) for csv_fold_path in results_csv_fold_paths]):
            CV_results = pd.concat(cv_results, axis=0, ignore_index=True)
            model_ROC = eval_.ROC_func(CV_results['Top-{}'.format(params['pred_window'])],CV_results['label'],max_pred=params['pred_window'],intervals=1000)
            np.save(os.path.join(split_path, '{}_ROC_'.format(model_type)+model_name), model_ROC)

            fig = plt.figure(figsize=(8, 6))
            tu.plot_roc('{}'.format(model_type),model_name,model_ROC)
            plt.legend()
            plt.savefig(os.path.join(split_path, '{}_ROC_'.format(model_type)+model_name+'.png'))
            plt.show()
            plt.close()

if __name__ == '__main__':
    import sys
    epochs = int(sys.argv[1])
    fold = int(sys.argv[2])
    lr = float(sys.argv[3])
    dataset = 'kits23_nooverlap'
    home = '/bask/projects/p/phwq4930-renal-canc/KCD_data/Data'
    train_cv_slice_model(home=home,dataname=dataset,splits=[0],train_folds=[fold],epochs=epochs,learning_rate=lr)
