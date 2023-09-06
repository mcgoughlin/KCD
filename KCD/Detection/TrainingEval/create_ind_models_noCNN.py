import KCD.Detection.Dataloaders.object_dataloader as dl_shapeimport eval_scripts as eval_import pandas as pdimport osimport torchimport torch.nn as nnimport model_generatorfrom torch.utils.data import DataLoaderfrom sklearn.model_selection import StratifiedKFold as kfold_stratimport numpy as npimport sysimport dglimport warningsimport gcif torch.cuda.is_available():    dev = 'cuda'else:    dev = 'cpu'def shape_collate(samples,dev=dev):    # The input `samples` is a list of pairs    #  (sv_im,features,graph, obj_label).    features,graphs, labels = map(list, zip(*samples))    batched_graph = dgl.batch(graphs)    return torch.stack(features).to(dev),batched_graph.to(dev), torch.stack(labels).squeeze(dim=0).to(dev)def compute_loss_weighting(spec_multiplier,num_classes=2):    sens = num_classes/(spec_multiplier+1)    spec = spec_multiplier*sens        return sens,spec#### individual model arch. paramstw_size = 'large'sv_size= 'small'gnn_layers = 5gnn_hiddendim = 25gnn_neighbours = 2#### individual model opt. paramsobject_batchsize = 8graph_thresh = 500mlp_thresh=20000mlp_lr = 0.01gnn_lr = 0.001#### ensemble model opt. paramss1_objepochs = 100#### sens/spec weighting during trainings1_spec_multiplier = 1.0s1_spec, s1_sens = compute_loss_weighting(s1_spec_multiplier)#### data paramstile_dataname = 'coreg_ncct'shape_ensemble_dataname ='combined_dataset_23andAdds'complete_ensemble_dataname = 'coreg_ncct'#### ignore all warnings - due to dgl being very annoyingignore = Trueif ignore: warnings.filterwarnings("ignore")#### path init  save_dir = '/bask/projects/p/phwq4930-renal-canc/EnsembleResults/saved_info'if not os.path.exists(save_dir):    os.mkdir(save_dir)obj_path = '/bask/projects/p/phwq4930-renal-canc/GraphData/'#### training housekeepingresults=[] five_fold_strat= kfold_strat(n_splits=5,shuffle=True)softmax = nn.Softmax(dim=-1)loss_fnc = nn.CrossEntropyLoss().to(dev)#### init all datasets needed shapedataset = dl_shape.EnsembleDataset(obj_path,                         data_name=shape_ensemble_dataname,                         graph_thresh=graph_thresh,mlp_thresh=mlp_thresh)test_shapedataset = dl_shape.EnsembleDataset(obj_path,                         data_name=shape_ensemble_dataname,                         graph_thresh=0,mlp_thresh=0)# highlight cases that are authentic ncct as 1, others as 0# this allows the k-fold case allocation to evenly split along the authentic ncct classcases = np.unique(shapedataset.cases.values)is_ncct = np.array([0 if case.startswith('case') else 1 for case in cases])test_res,train_res = [], []cv_results = []for reading in range(1):    for split in [0]:        split_path = os.path.join(save_dir,'split_{}'.format(split))        if not os.path.exists(split_path):            os.mkdir(split_path)                    split_fp = os.path.join(split_path,'split.npy')        if os.path.exists(split_fp):            fold_split = np.load(split_fp,allow_pickle=True)        else:              fold_split = np.array([(fold,tr_index,ts_index) for fold,(tr_index, ts_index) in enumerate(five_fold_strat.split(cases,is_ncct))])            np.save(os.path.join(split_path,split_fp),fold_split)                    # begin training!        for fold,train_index, test_index in fold_split:             fold_path = os.path.join(split_path,'fold_{}'.format(fold))            MLP_path = os.path.join(fold_path,'MLP')            svCNN_path = os.path.join(fold_path,'svCNN')            twCNN_path = os.path.join(fold_path,'twCNN')            GNN_path = os.path.join(fold_path,'GNN')                        if not os.path.exists(fold_path):                os.mkdir(fold_path)                os.mkdir(MLP_path)                os.mkdir(svCNN_path)                os.mkdir(twCNN_path)                os.mkdir(GNN_path)                            # extract ncct cases from the fold split,             # use this later to fine-tune only on authentic ncct.            ncct_cases = np.array([case for case in cases[train_index] if not case.startswith('case')])            ncct_cases = np.array([case if not case.startswith('KiTS') else case.replace('-','_') for case in ncct_cases ])                    # init models            MLP = model_generator.return_MLP(dev=dev)            print(MLP)            GNN = model_generator.return_graphnn(num_features=4,num_labels=2,layers_deep=gnn_layers,hidden_dim=gnn_hiddendim,neighbours=gnn_neighbours,dev=dev)                    print("\nFold {} training.".format(fold))                        ######## Individual Object Model Training ########            print("Training object classifiers")                        GNNopt = torch.optim.Adam(GNN.parameters(),lr=gnn_lr)            MLPopt = torch.optim.Adam(MLP.parameters(),lr=mlp_lr)                    shapedataset.apply_foldsplit(train_cases = cases[train_index])            test_shapedataset.apply_foldsplit(train_cases = cases[train_index])            dl = DataLoader(shapedataset,batch_size=object_batchsize,shuffle=True,collate_fn=shape_collate)            test_dl = DataLoader(test_shapedataset,batch_size=object_batchsize,shuffle=True,collate_fn=shape_collate)                        loss_fnc.weight = torch.Tensor([s1_spec,s1_sens]).to(dev)            MLP.train(),GNN.train()            for i in range(s1_objepochs):                print("\nEpoch {}".format(i))                for features,graph, lb in dl:                    feat_lb,graph_lb = lb.T                    MLPpred = MLP(features.to(dev))                    GNNpred = GNN(graph.to(dev))                    for pred,opt,label in zip([MLPpred,GNNpred],[MLPopt,GNNopt], [feat_lb,graph_lb]):                        loss = loss_fnc(pred,label.to(dev))                        loss.backward()                        opt.step()                        opt.zero_grad()                                    for opt in [MLPopt,GNNopt]:                opt.zero_grad()                del(opt)                            MLP_name = '{}_{}_{}_{}_{}_{}'.format(s1_objepochs,                                                    mlp_thresh,                                                    mlp_lr,                                                    s1_spec_multiplier,                                                    object_batchsize,                                                    reading)            GNN_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(s1_objepochs,                                                            graph_thresh,                                                            gnn_lr,                                                            gnn_layers,                                                            gnn_hiddendim,                                                            gnn_neighbours,                                                            s1_spec_multiplier,                                                            object_batchsize,                                                            reading)                        torch.save(MLP,os.path.join(MLP_path,'model',MLP_name))            torch.save(GNN,os.path.join(GNN_path,'model',GNN_name))                        GNN.eval(),MLP.eval()            shape_model_res,test_df = eval_.eval_shape_models(GNN,MLP,test_dl,dev=dev)            shape_model_res = shape_model_res.reset_index(level=['model'])            cv_results.append(test_df)                        GNN_res = shape_model_res[shape_model_res['model']=='GNN'].drop('model',axis=1)            MLP_res = shape_model_res[shape_model_res['model']=='MLP'].drop('model',axis=1)                        GNN_res.to_csv(os.path.join(GNN_path,'csv',MLP_name+'.csv'))            MLP_res.to_csv(os.path.join(MLP_path,'csv',GNN_name+'.csv'))            CV_results = pd.concat(cv_results, axis=0, ignore_index=True)GNN_ROC = eval_.ROC_func(CV_results['GNNpred'],CV_results['label'],max_pred=1,intervals=1000)MLP_ROC = eval_.ROC_func(CV_results['MLPpred'],CV_results['label'],max_pred=1,intervals=1000)np.save(os.path.join(split_path,'MLP_ROC'),MLP_ROC)np.save(os.path.join(split_path,'GNN_ROC'),GNN_ROC)print(GNN_ROC)            