import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ROC_func(pred_var,lab,max_pred,intervals=20):
    boundaries = np.arange(-0.01,max_pred+0.01,1/intervals)
    # pred [malig,non-malig]
    sens_spec = []
    is_truely_malig = lab==1
    for boundary in boundaries:
        # pred is zero if benign, 1 if malig
        new_pred = pred_var> boundary
        correct = new_pred == is_truely_malig

        sens = (correct & new_pred).sum()/(is_truely_malig).sum()
        spec = (correct & ~new_pred).sum()/(~is_truely_malig).sum()

        sens_spec.append([sens,spec])
    
    return np.array(sens_spec,dtype=float)

def eval_twcnn(twCNN,test_tw_dl,ps_boundary=0.98,dev='cpu',boundary_size=10):
    boundary =ps_boundary*boundary_size
    twCNN.eval()
    test_res = []
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        test_tw_dl.dataset.is_train=True
        for case in test_tw_dl.dataset.cases:
            for position in test_tw_dl.dataset.data_df[test_tw_dl.dataset.data_df['case'] ==case].side.unique():
                if position == 'random': continue
                test_tw_dl.dataset.set_val_kidney(case,position)
                entry = {'case':case,
                         'position':position}
                case_store = []
                for x,lb in test_tw_dl:
                    pred = softmax(twCNN(x.to(dev)))
                    case_store.extend(pred[:,2].cpu().numpy().tolist())

                case_store.sort(reverse=True)
                print(case, position,sum(case_store[:boundary_size]),boundary)
                entry['Top-{}'.format(boundary_size)]=sum(case_store[:boundary_size])
                entry['prediction'] = int(entry['Top-{}'.format(boundary_size)]>=boundary)
                entry['boundary']=boundary
                test_res.append(entry)
            
    test_df = pd.DataFrame(test_res)
    
    return test_df.drop_duplicates()



def eval_shape_ensemble(shape_ensemble,test_dl,boundary=0.98,dev='cpu'):
    shape_ensemble.eval()
    shape_ensemble.MLP.eval(), shape_ensemble.GNN.eval(), shape_ensemble.CNN.eval()
    test_res = []
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        test_dl.dataset.is_train=True
        cases = np.unique(test_dl.dataset.cases)
        print('#############')
        print(cases)
        for case in cases:
            for position in test_dl.dataset.case_data[test_dl.dataset.case_data['case'] ==case].position.unique():
                test_dl.dataset.set_val_kidney(case,position)
                entry = {'case':case,
                         'position':position}
                print(entry)
                for sv_im,features,graph, lb in test_dl:
                    sv_lb,feat_lb,graph_lb = lb.T
                    pred = softmax(shape_ensemble(features,sv_im,graph))
                    
                entry['pred-cont']=pred[0,1].item()
                print(pred[0,1].item())
                entry['pred-hard'] = int(entry['pred-cont']>=boundary)
                entry['boundary']=boundary
                test_res.append(entry)
            
    test_df = pd.DataFrame(test_res)
    
    return test_df.drop_duplicates()

def eval_shape_ensemble_nocnn(shape_ensemble,test_dl,boundary=0.98,dev='cpu'):
    shape_ensemble.eval()
    shape_ensemble.MLP.eval(), shape_ensemble.GNN.eval()
    test_res = []
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        test_dl.dataset.is_train=True
        cases = np.unique(test_dl.dataset.cases)
        print('#############')
        print(cases)
        print(dev)
        print(shape_ensemble.device)
        for case in cases:
            for position in test_dl.dataset.case_data[test_dl.dataset.case_data['case'] ==case].position.unique():
                test_dl.dataset.set_val_kidney(case,position)
                entry = {'case':case,
                         'position':position}
                print(entry)
                for features,graph, lb in test_dl:
                    feat_lb,graph_lb = lb.T
                    pred = softmax(shape_ensemble(features,graph))
                    
                entry['pred-cont']=pred[0,1].item()
                print(pred[0,1].item())
                entry['pred-hard'] = int(entry['pred-cont']>=boundary)
                entry['boundary']=boundary
                test_res.append(entry)
            
    test_df = pd.DataFrame(test_res)
    
    return test_df.drop_duplicates()

