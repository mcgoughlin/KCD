import torch
import os
os.environ['OV_DATA_BASE'] ='/bask/projects/p/phwq4930-renal-canc/data/seg_data'
from torch.utils.data import DataLoader

from KCD.ne2ceCT.dataloader import CrossPhaseDataset
from KCD.ne2ceCT.loss_function import PyramidalLatentSimilarityLoss
from KCD.Segmentation.Inference import infer_network
from KCD.ne2ceCT.lr_scheduler import LinearWarmupExponentialDecayScheduler
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import sys

def get_3d_featureoutput_unet(in_channels, out_channels, n_stages,filters=32, filters_max=1024):
    kernel_sizes = n_stages * [(3, 3, 3)]

    return infer_network.UNet_featureoutput(in_channels, out_channels, kernel_sizes, False, filters_max=filters_max, filters=filters, )

spacing = 2
batch_size = 40
epochs = int(sys.argv[1])
l1_weight = float(sys.argv[2])
l2_weight = float(sys.argv[3])
cce_weight = float(sys.argv[4])
cos_weight = float(sys.argv[5])
var_loss_weight = float(sys.argv[6])
cov_loss_weight = float(sys.argv[7])
lr_max = float(sys.argv[8])
loss_gamma = float(sys.argv[9])

if __name__ == '__main__':

    dataset_ne_path = os.path.join(os.environ['OV_DATA_BASE'],'preprocessed/small_coreg_ncct/small_coreg_ncct_{}/'.format(spacing))
    dataset_ce_path = os.path.join(os.environ['OV_DATA_BASE'],('preprocessed/add_cect/add_cect_{}/'.format(spacing)))

    # dataset_ne_path = os.path.join(home_path,'preprocessed', 'small_coreg_ncct','{}mm_allbinary'.format(spacing))
    # dataset_ce_path = os.path.join(home_path,'preprocessed', 'small_coreg_ncct','{}mm_allbinary'.format(spacing))

    assert os.path.exists(dataset_ne_path) and os.path.exists(dataset_ce_path)
    ne2ceCT_path = os.path.join(os.environ['OV_DATA_BASE'], 'ne2ceCT','small_coreg_alllabel_{}'.format(spacing), 'l1{}_l2{}_cce{}_cos{}_{}lr_{}lg_{}bs_{}ep'.format(l1_weight,
                                                                                                                  l2_weight,
                                                                                                                  cce_weight,
                                                                                                                  cos_weight,
                                                                                                                  lr_max,
                                                                                                                  loss_gamma,
                                                                                                                  batch_size,
                                                                                                                  epochs))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_path = os.path.join(ne2ceCT_path, 'logs.txt')

    if not os.path.exists(ne2ceCT_path):
        os.makedirs(ne2ceCT_path)


    if os.path.exists(log_path):
        os.remove(log_path)


    dataset = CrossPhaseDataset(os.path.join(dataset_ne_path,'images'), os.path.join(dataset_ce_path,'images'),
                                device=dev,
                                is_train=True,patches_per_case=40)

    dataset.apply_foldsplit(split_ratio=0.9)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=5)


    model_T = get_3d_featureoutput_unet(1, 4, 6,  filters=32, filters_max=1024).to(dev)
    model_T.load_state_dict(torch.load('/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/kits23_nooverlap/2mm_alllabel/alllabel_long/fold_0/network_weights'))
    model_S = get_3d_featureoutput_unet(1, 4, 6,  filters=32, filters_max=1024).to(dev)
    model_S.load_state_dict(torch.load('/bask/projects/p/phwq4930-renal-canc/data/seg_data/trained_models/kits23_nooverlap/2mm_alllabel/alllabel_long/fold_0/network_weights'))

    model_T.eval()
    model_S.train()

    optimizer = torch.optim.Adam(model_S.parameters(), lr=lr_max/20, betas=(0.95, 0.9), eps=1e-08, weight_decay=0.0001)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=loss_gamma)

    scheduler = LinearWarmupExponentialDecayScheduler(optimizer, warmup_epochs=500, lr_max=lr_max,
                                                      decay_rate=loss_gamma)

    loss_func = PyramidalLatentSimilarityLoss(l1_weight=l1_weight, l2_weight=l2_weight,
                                              cce_weight=cce_weight, cos_weight=cos_weight,
                                              var_loss_weight=var_loss_weight,cov_loss_weight=cov_loss_weight).to(dev)
    losses = []
    weight = 0.95
    running_loss = None
    running_val_loss = None
    min_val_loss = 1e6
    for epoch in range(epochs):
        dataloader.dataset.is_train = True
        model_S.train()
        for i, (ne, ce) in enumerate(dataloader):
            # forward
            ne = ne.to(dev)
            ce = ce.to(dev)
            with torch.no_grad():
                T = model_T(ce)
            S = model_S(ne)
            loss = loss_func(T, S)
            # backward
            loss.backward()
            # update weights
            optimizer.step()
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = weight * running_loss + (1 - weight) * loss.item()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, len(dataloader), loss.item()))
            #print like above but with line replacement

            losses.append(running_loss)
            scheduler.step()
        train_string = f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss:.4E}, LR: {scheduler.get_last_lr()[0]:.4E}\n'
        with open(log_path, 'a') as f:
            f.write(train_string)

        dataloader.dataset.is_train = False
        with torch.no_grad():
            model_S.eval()
            for i, (ne, ce) in enumerate(dataloader):
                ne = ne.to(dev)
                ce = ce.to(dev)
                T = model_T(ce)
                S = model_S(ne)
                loss = loss_func(T, S)
                if running_val_loss is None:
                    running_val_loss = loss.item()
                else:
                    running_val_loss = weight * running_val_loss + (1 - weight) * loss.item()

        val_string = f'Val Loss: {running_val_loss:.4E}\n'
        with open(log_path, 'a') as f:
            f.write(val_string)
        dataloader.dataset.reset_patch_count()
        if running_val_loss < min_val_loss:
            min_val_loss = running_val_loss
            torch.save(model_S.state_dict(), os.path.join(ne2ceCT_path, 'network_weights'))
            print('Model saved')
            with open(log_path, 'a') as f:
                f.write('Model saved\n')

        with open(log_path, 'a') as f:
            f.write('\n')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(losses)
        plt.savefig(os.path.join(ne2ceCT_path, 'losses.png'))
