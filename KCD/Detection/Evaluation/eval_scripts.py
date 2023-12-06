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

def eval_cnn(CNN,test_tw_dl,plot_path=None,dev='cpu'):
    
    train_res,test_res,final_results = [],[],[]
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
<<<<<<< HEAD
        for case in np.unique(test_tw_dl.dataset.train_cases.values):
=======
        for case in np.unique(test_tw_dl.dataset.train_cases):
>>>>>>> 43b3b81ff9ef7b552b3d71fbb7c3fa646b5fdbc1
            for position in test_tw_dl.dataset.data_df[test_tw_dl.dataset.data_df['case'] ==case].side.unique():
                if position == 'random': continue
                test_tw_dl.dataset.set_val_kidney(case,position)
                entry = {'case':case,
                         'position':position}
                print(case,position)
                label = 0
                case_store = []
                for x,lb in test_tw_dl:
                    if any(lb==2): label = 1
                    pred = softmax(CNN(x.to(dev)))
                    case_store.extend(pred[:,2].cpu().numpy().tolist())

                case_store.sort(reverse=True)
                entry['label']=label
                entry['Top-1']=case_store[0]
                entry['Top-3']=sum(case_store[:3])
                entry['Top-5']=sum(case_store[:5])
                entry['Top-10']=sum(case_store[:10])
                entry['Top-15']=sum(case_store[:15])
                entry['Top-20']=sum(case_store[:20])
                train_res.append(entry)
    
        test_tw_dl.dataset.is_train=False
<<<<<<< HEAD
        for case in np.unique(test_tw_dl.dataset.test_cases.values):
=======
        for case in np.unique(test_tw_dl.dataset.test_cases):
>>>>>>> 43b3b81ff9ef7b552b3d71fbb7c3fa646b5fdbc1
            for position in test_tw_dl.dataset.data_df[test_tw_dl.dataset.data_df['case'] ==case].side.unique():
                if position == 'random': continue
                test_tw_dl.dataset.set_val_kidney(case,position)
                entry = {'case':case,
                         'position':position}

                print(case,position)
                label = 0
                case_store = []
                for x,lb in test_tw_dl:
                    if any(lb==2): label = 1
                    pred = softmax(CNN(x.to(dev)))
                    case_store.extend(pred[:,2].cpu().numpy().tolist())
                    
                case_store.sort(reverse=True)
                entry['label']=label
                entry['Top-1']=case_store[0]
                entry['Top-3']=sum(case_store[:3])
                entry['Top-5']=sum(case_store[:5])
                entry['Top-10']=sum(case_store[:10])
                entry['Top-15']=sum(case_store[:15])
                entry['Top-20']=sum(case_store[:20])
                test_res.append(entry)
            
    train_df = pd.DataFrame(train_res)
    test_df = pd.DataFrame(test_res)
    
    for df, df_name in zip([train_df,test_df],['train','test']):
        for col in [column for column in df.columns if (('Top-' in column) or ('Neighbourhood' in column))]:
            max_pred = int(col.split('-')[1])
            type_ = col.split('-')[0]
            pred = df[col].values
            label = df.label.values
                    
            ens_ROC= ROC_func(pred,label,max_pred=max_pred,intervals=1000)
            AUC = np.trapz(ens_ROC[:,0],ens_ROC[:,1])
            
            
            sens98spec=ens_ROC[ens_ROC[:,1]>0.98][:,0].max()
            sens95spec=ens_ROC[ens_ROC[:,1]>0.95][:,0].max()
            sens90spec=ens_ROC[ens_ROC[:,1]>0.90][:,0].max()
            sens100spec=ens_ROC[ens_ROC[:,1]==1.0][:,0].max()
            
            boundary90 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens90spec])/len(ens_ROC)
            boundary95 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens95spec])/len(ens_ROC)
            boundary98 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens98spec])/len(ens_ROC)
            boundary100 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens100spec])/len(ens_ROC)
    
                    
            final_results.append({'dataset_loc':df_name,
                                  'Type':type_,
                                  'Voting Size':max_pred,
                    'AUC':AUC,
                    'Highest Cancer Sens @ 100% Cancer Spec':sens100spec,
                    'Highest Cancer Sens @ 98% Cancer Spec':sens98spec,
                    'Highest Cancer Sens @ 95% Cancer Spec':sens95spec,
                    'Highest Cancer Sens @ 90% Cancer Spec':sens90spec,
                    'Boundary 90':boundary90,
                    'Boundary 95':boundary95,
                    'Boundary 98':boundary98,
                    'Boundary 100':boundary100})

    print('FINISHED')
    results_df = pd.DataFrame(final_results).groupby(['Type','Voting Size','dataset_loc']).mean()
    return results_df,test_df


def eval_shape_ensemble(shape_ensemble,test_dl,plot_path=None,dev='cpu'):
    test_dl.dataset.is_train=True
    train_res,test_res,final_results = [],[],[]
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        for case in test_dl.dataset.train_cases:
            for position in test_dl.dataset.case_data[test_dl.dataset.case_data['case'] ==case].position.unique():
                test_dl.dataset.set_val_kidney(case,position)
                entry = {'case':case,
                         'position':position}
                for features,graph, lb in test_dl:
                    pred = softmax(shape_ensemble(features,graph))

                entry['label']=lb.item()
                entry['pred']=pred[0,1].item()

                train_res.append(entry)

        test_dl.dataset.is_train=False
        for case in test_dl.dataset.test_cases:
            for position in test_dl.dataset.case_data[test_dl.dataset.case_data['case'] ==case].position.unique():
                test_dl.dataset.set_val_kidney(case,position)
                entry = {'case':case,
                         'position':position}
                for features,graph, lb in test_dl:
                    pred = softmax(shape_ensemble(features,graph))
                    
                entry['label']=lb.item()
                entry['pred']=pred[0,1].item()
                test_res.append(entry)
            
    train_df = pd.DataFrame(train_res)
    test_df = pd.DataFrame(test_res)
    
    for df, df_name in zip([train_df,test_df],['train','test']):
        pred = df.pred.values
        label = df.label.values
                
        ens_ROC= ROC_func(pred,label,max_pred=1,intervals=1000)
        AUC = np.trapz(ens_ROC[:,0],ens_ROC[:,1])
        
        sens98spec=ens_ROC[ens_ROC[:,1]>0.98][:,0].max()
        sens95spec=ens_ROC[ens_ROC[:,1]>0.95][:,0].max()
        sens90spec=ens_ROC[ens_ROC[:,1]>0.90][:,0].max()
        sens100spec=ens_ROC[ens_ROC[:,1]==1.0][:,0].max()
        
        boundary90 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens90spec])/len(ens_ROC)
        boundary95 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens95spec])/len(ens_ROC)
        boundary98 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens98spec])/len(ens_ROC)
        boundary100 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens100spec])/len(ens_ROC)

        final_results.append({'dataset_loc':df_name,
                'AUC':AUC,
                'Highest Cancer Sens @ 100% Cancer Spec':sens100spec,
                'Highest Cancer Sens @ 98% Cancer Spec':sens98spec,
                'Highest Cancer Sens @ 95% Cancer Spec':sens95spec,
                'Highest Cancer Sens @ 90% Cancer Spec':sens90spec,
                'Boundary 90':boundary90,
                'Boundary 95':boundary95,
                'Boundary 98':boundary98,
                'Boundary 100':boundary100})

    results_df = pd.DataFrame(final_results).groupby(['dataset_loc']).mean()
    return results_df,test_df


def eval_shape_models_xgb(GNN, MLP,xgb,
                          test_dl, plot_path=None, dev='cpu'):
    test_dl.dataset.is_train = True
    train_res, test_res, final_results = [], [], []
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        for case in test_dl.dataset.train_cases:
            for position in test_dl.dataset.case_data[test_dl.dataset.case_data['case'] == case].position.unique():
                test_dl.dataset.set_val_kidney(case, position)
                entry = {'case': case,
                         'position': position}
                for features, graph, lb in test_dl:
                    feat_lb, graph_lb = lb.T
                    GNNpred = softmax(GNN(graph))
                    XGBpred = softmax(torch.Tensor(xgb.predict_proba(features.detach().cpu().numpy())))
                    MLPpred = softmax(MLP(features))

                entry['label'] = feat_lb.item()
                entry['GNNpred'] = GNNpred[0, 1].item()
                entry['MLPpred'] = MLPpred[0, 1].item()
                entry['XGBpred'] = XGBpred[0, 1].item()

                train_res.append(entry)

        test_dl.dataset.is_train = False
        for case in test_dl.dataset.test_cases:
            for position in test_dl.dataset.case_data[test_dl.dataset.case_data['case'] == case].position.unique():
                test_dl.dataset.set_val_kidney(case, position)
                entry = {'case': case,
                         'position': position}
                for features, graph, lb in test_dl:
                    feat_lb, graph_lb = lb.T
                    GNNpred = softmax(GNN(graph))
                    XGBpred = softmax(torch.Tensor(xgb.predict_proba(features.detach().cpu().numpy())))
                    MLPpred = softmax(MLP(features))

                entry['label'] = feat_lb.item()
                entry['GNNpred'] = GNNpred[0, 1].item()
                entry['MLPpred'] = MLPpred[0, 1].item()
                entry['XGBpred'] = XGBpred[0, 1].item()
                test_res.append(entry)

    train_df = pd.DataFrame(train_res)
    test_df = pd.DataFrame(test_res)

    for df, df_name in zip([train_df, test_df], ['train', 'test']):
        GNNpred = df.GNNpred.values
        MLPpred = df.MLPpred.values
        XGBpred = df.XGBpred.values
        label = df.label.values

        GNNens_ROC = ROC_func(GNNpred, label, max_pred=1, intervals=1000)
        MLPens_ROC = ROC_func(MLPpred, label, max_pred=1, intervals=1000)
        XGBens_ROC = ROC_func(XGBpred, label, max_pred=1, intervals=1000)

        for model_name, ens_ROC in zip(['GNN', 'MLP', 'XGB'], [GNNens_ROC, MLPens_ROC,XGBens_ROC]):
            AUC = np.trapz(ens_ROC[:, 0], ens_ROC[:, 1])

            sens98spec = ens_ROC[ens_ROC[:, 1] > 0.98][:, 0].max()
            sens95spec = ens_ROC[ens_ROC[:, 1] > 0.95][:, 0].max()
            sens90spec = ens_ROC[ens_ROC[:, 1] > 0.90][:, 0].max()
            sens100spec = ens_ROC[ens_ROC[:, 1] == 1.0][:, 0].max()

            boundary90 = max([i for i, (sens, spec) in enumerate(ens_ROC) if sens == sens90spec]) / len(ens_ROC)
            boundary95 = max([i for i, (sens, spec) in enumerate(ens_ROC) if sens == sens95spec]) / len(ens_ROC)
            boundary98 = max([i for i, (sens, spec) in enumerate(ens_ROC) if sens == sens98spec]) / len(ens_ROC)
            boundary100 = max([i for i, (sens, spec) in enumerate(ens_ROC) if sens == sens100spec]) / len(ens_ROC)

            final_results.append({'dataset_loc': df_name,
                                  'AUC': AUC,
                                  'model': model_name,
                                  'Highest Cancer Sens @ 100% Cancer Spec': sens100spec,
                                  'Highest Cancer Sens @ 98% Cancer Spec': sens98spec,
                                  'Highest Cancer Sens @ 95% Cancer Spec': sens95spec,
                                  'Highest Cancer Sens @ 90% Cancer Spec': sens90spec,
                                  'Boundary 90': boundary90,
                                  'Boundary 95': boundary95,
                                  'Boundary 98': boundary98,
                                  'Boundary 100': boundary100})

    results_df = pd.DataFrame(final_results).groupby(['dataset_loc', 'model']).mean()
    return results_df, test_df

def eval_shape_models(GNN,MLP,test_dl,plot_path=None,dev='cpu'):
    test_dl.dataset.is_train=True
    train_res,test_res,final_results = [],[],[]
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        for case in test_dl.dataset.train_cases:
            for position in test_dl.dataset.case_data[test_dl.dataset.case_data['case'] ==case].position.unique():
                test_dl.dataset.set_val_kidney(case,position)
                entry = {'case':case,
                         'position':position}
                for features,graph, lb in test_dl:
                    feat_lb,graph_lb = lb.T
                    GNNpred = softmax(GNN(graph))
                    MLPpred = softmax(MLP(features))

                entry['label']=feat_lb.item()
                entry['GNNpred']=GNNpred[0,1].item()
                entry['MLPpred']=MLPpred[0,1].item()

                train_res.append(entry)

        test_dl.dataset.is_train=False
        for case in test_dl.dataset.test_cases:
            for position in test_dl.dataset.case_data[test_dl.dataset.case_data['case'] ==case].position.unique():
                test_dl.dataset.set_val_kidney(case,position)
                entry = {'case':case,
                         'position':position}
                for features,graph, lb in test_dl:
                    feat_lb,graph_lb = lb.T
                    GNNpred = softmax(GNN(graph))
                    MLPpred = softmax(MLP(features))

                entry['label']=feat_lb.item()
                entry['GNNpred']=GNNpred[0,1].item()
                entry['MLPpred']=MLPpred[0,1].item()
                test_res.append(entry)
            
    train_df = pd.DataFrame(train_res)
    test_df = pd.DataFrame(test_res)
    
    for df, df_name in zip([train_df,test_df],['train','test']):
        GNNpred = df.GNNpred.values
        MLPpred = df.MLPpred.values
        label = df.label.values
                
        GNNens_ROC= ROC_func(GNNpred,label,max_pred=1,intervals=1000)
        MLPens_ROC= ROC_func(MLPpred,label,max_pred=1,intervals=1000)
        
        for model_name,ens_ROC in zip(['GNN','MLP'],[GNNens_ROC,MLPens_ROC]):
            AUC = np.trapz(ens_ROC[:,0],ens_ROC[:,1])
            
            sens98spec=ens_ROC[ens_ROC[:,1]>0.98][:,0].max()
            sens95spec=ens_ROC[ens_ROC[:,1]>0.95][:,0].max()
            sens90spec=ens_ROC[ens_ROC[:,1]>0.90][:,0].max()
            sens100spec=ens_ROC[ens_ROC[:,1]==1.0][:,0].max()
            
            boundary90 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens90spec])/len(ens_ROC)
            boundary95 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens95spec])/len(ens_ROC)
            boundary98 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens98spec])/len(ens_ROC)
            boundary100 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens100spec])/len(ens_ROC)
    
                    
            final_results.append({'dataset_loc':df_name,
                    'AUC':AUC,
                    'model':model_name,
                    'Highest Cancer Sens @ 100% Cancer Spec':sens100spec,
                    'Highest Cancer Sens @ 98% Cancer Spec':sens98spec,
                    'Highest Cancer Sens @ 95% Cancer Spec':sens95spec,
                    'Highest Cancer Sens @ 90% Cancer Spec':sens90spec,
                    'Boundary 90':boundary90,
                    'Boundary 95':boundary95,
                    'Boundary 98':boundary98,
                    'Boundary 100':boundary100})

    results_df = pd.DataFrame(final_results).groupby(['dataset_loc','model']).mean()
    return results_df,test_df

def eval_kwcnn(kwcnn,tilestack_dl,dev='cpu'):
    tilestack_dl.dataset.is_train=True
    train_res,test_res,final_results = [],[],[]
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        for case in tilestack_dl.dataset.train_cases:
            for position in tilestack_dl.dataset.tile_df[tilestack_dl.dataset.tile_df['case'] ==case].side.unique():
                tilestack_dl.dataset.set_val_kidney(case,position)
                entry = {'case':case,
                         'position':position}
                for tile_ims, lb in tilestack_dl:
                    pred = softmax(kwcnn(tile_ims))

                entry['label']=lb.item()
                entry['pred']=pred[0,1].item()
                print(entry)
                train_res.append(entry)

        tilestack_dl.dataset.is_train=False
        for case in tilestack_dl.dataset.test_cases:
            for position in tilestack_dl.dataset.tile_df[tilestack_dl.dataset.tile_df['case'] ==case].side.unique():
                tilestack_dl.dataset.set_val_kidney(case,position)
                entry = {'case':case,
                         'position':position}
                for tile_ims, lb in tilestack_dl:
                    pred = softmax(kwcnn(tile_ims))

                entry['label']=lb.item()
                entry['pred']=pred[0,1].item()
                print(entry)
                test_res.append(entry)
            
    train_df = pd.DataFrame(train_res)
    test_df = pd.DataFrame(test_res)
    
    for df, df_name in zip([train_df,test_df],['train','test']):
        print(df)
        print(df.columns)
        pred = df.pred.values
        label = df.label.values
                
        ens_ROC= ROC_func(pred,label,max_pred=1,intervals=1000)
        AUC = np.trapz(ens_ROC[:,0],ens_ROC[:,1])
        
        
        sens98spec=ens_ROC[ens_ROC[:,1]>0.98][:,0].max()
        sens95spec=ens_ROC[ens_ROC[:,1]>0.95][:,0].max()
        sens90spec=ens_ROC[ens_ROC[:,1]>0.90][:,0].max()
        sens100spec=ens_ROC[ens_ROC[:,1]==1.0][:,0].max()
        
        boundary90 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens90spec])/len(ens_ROC)
        boundary95 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens95spec])/len(ens_ROC)
        boundary98 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens98spec])/len(ens_ROC)
        boundary100 = max([i for i, (sens,spec) in enumerate(ens_ROC) if sens==sens100spec])/len(ens_ROC)

                
        final_results.append({'dataset_loc':df_name,
                'AUC':AUC,
                'Highest Cancer Sens @ 100% Cancer Spec':sens100spec,
                'Highest Cancer Sens @ 98% Cancer Spec':sens98spec,
                'Highest Cancer Sens @ 95% Cancer Spec':sens95spec,
                'Highest Cancer Sens @ 90% Cancer Spec':sens90spec,
                'Boundary 90':boundary90,
                'Boundary 95':boundary95,
                'Boundary 98':boundary98,
                'Boundary 100':boundary100})


        
    results_df = pd.DataFrame(final_results).groupby(['dataset_loc']).mean()
    return results_df

