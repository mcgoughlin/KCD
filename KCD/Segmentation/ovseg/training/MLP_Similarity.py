import os
os.environ['OV_DATA_BASE'] = "/Users/mcgoug01/Downloads/ovseg_test"
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MLP_Similarity(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP_Similarity, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, input_size)]
        )
        self.skip = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.Linear(input_size, hidden_size)])

        self.dropout = nn.Dropout(0.9)

    def forward(self, input):

        dec0 = F.tanh(self.dropout(self.encoder[0](input)))
        dec1 = F.tanh(self.dropout(self.encoder[1](dec0)))
        dec2 = F.tanh(self.dropout(self.encoder[2](dec1)+F.tanh(self.skip[1](input))))
        x = self.encoder[3](dec2+F.tanh(self.skip[0](input)))
        return x

class MLP_encdec(nn.Module):
    def __init__(self, input_size, hidden_size, bottom_size):
        super(MLP_encdec, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, bottom_size)]
        )
        self.decoder = nn.ModuleList([
            nn.Linear(bottom_size,hidden_size),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, input_size)]
        )

    def forward(self, x,calc_sim=True,sim_MLP=None,sim_loss=None,
                sim_opt=None, print_sim=False):
        encodings = []
        for layer in self.encoder:
            x = F.tanh(layer(x))
            encodings.append(x)


        x = F.tanh(self.decoder[0](x))
        if calc_sim:
            ts_loss,sim_MLP = train_sim_model(sim_MLP,x,encodings[1],
                                      sim_loss,sim_opt)

        x = torch.cat([x, encodings[1]], dim=1)
        x = F.tanh(self.decoder[1](x))
        x= F.tanh(self.decoder[2](x))
        if calc_sim:
            return self.decoder[3](x), ts_loss, sim_MLP
        else:
            return self.decoder[3](x)

def train_sim_model(sim_MLP, x, encoding,sim_loss_func,opt):
    sim_MLP.train()
    sim_MLP = sim_MLP
    # calculate the MI between the two representations
    shuffle_index = torch.randperm(x.shape[0])
    x_, enc_ = x[shuffle_index].clone().detach(), encoding[shuffle_index]
    x_tr, x_test = x_[:x_.shape[0] // 5], x_[x_.shape[0] // 5:]
    enc_tr, enc_test = enc_[:enc_.shape[0] // 5], enc_[enc_.shape[0] // 5:]
    for i in range(20):
        opt.zero_grad()
        enc_tr_pred = sim_MLP(x_tr.clone().detach())
        sim_loss = sim_loss_func(enc_tr_pred, enc_tr.clone().detach())
        sim_loss.backward()
        opt.step()
        if i ==0:
            tr_loss = sim_loss.item()
        opt.zero_grad()

    model.eval()
    enc_ts_pred = sim_MLP(x_test)
    ts_loss = sim_loss_func(enc_ts_pred, enc_test)
    return ts_loss,sim_MLP

if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)
    dims = 100
    hidden_size = 100
    bottom_size = hidden_size//2
    ds_size = 20000

    model = MLP_encdec(dims, hidden_size, bottom_size).to('cuda')
    optimizer = torch.optim.Adam(list(model.encoder.parameters())+
                                 list(model.decoder.parameters()),lr=5e-4)

    sim_model = MLP_Similarity(hidden_size,25).to('cuda')
    sim_opt = torch.optim.Adam(sim_model.parameters(), lr=1e-3)

    _ = np.random.rand(dims, dims)
    cov = np.tril(_) + np.tril(_, -1).T
    # correlated distribution
    data = np.random.multivariate_normal( mean=np.zeros(dims),
                                  cov=cov**2,
                                 size = ds_size).astype(np.float32)

    #add non-linearity
    data = data + 0.1*np.sin(data) + 0.1*np.exp(data) + 0.01*np.exp(np.sin(data)) + 0.001*(data**2) + 0.0001*(data**3)
    data = np.log10(np.exp(data))

    #add noise
    data += np.random.normal(0,2,data.shape)

    #replace any inf with normal distribution around the value 10
    data[np.isinf(data)] = np.random.normal(0,0.1,data[np.isinf(data)].shape)
    data[np.isinf(data)] = np.random.normal(data.max(),data.max()/10,data[np.isinf(data)].shape)

    #normalise the data
    data /= data.max()

    train_data,test_data = torch.utils.data.random_split(data,[8*ds_size//10,2*ds_size//10])

    #create a dataloader that returns a batch of data, input is identical to output
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=100,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=100,shuffle=True)

    #train the model
    criterion = nn.MSELoss().to('cuda')
    epochs = 100
    mse_losses = []
    sim_losses = []
    lambda_sim = 1e-3
    calc_sim = True
    include_sim = True

    for epoch in range(epochs):
        print_sim= True
        for batch in train_loader:
            batch = batch.to('cuda')
            optimizer.zero_grad()
            if calc_sim:
                outputs,dissim,sim_model = model(batch,calc_sim=calc_sim,print_sim=print_sim,sim_opt=sim_opt,
                                    sim_MLP=sim_model,sim_loss = criterion)
                sim = (1 - torch.exp(-dissim))
                sim_losses.append(sim.detach().to('cuda').item())
                MSE_loss = criterion(outputs, batch)
                loss = MSE_loss
                if include_sim:
                    loss += lambda_sim*sim
            else:
                outputs = model(batch, calc_sim=False)
                MSE_loss = criterion(outputs, batch)
                loss = MSE_loss
            mse_losses.append(MSE_loss.detach().to('cuda').item())
            loss.backward()
            optimizer.step()
            print_sim = False
        if calc_sim:
            print('epoch {}, end2end loss {:.6f}, PixPred Similarity {:.6f}, Total Loss {:.6f}'.format(epoch,
                                                                                                MSE_loss.item(),
                                                                                                sim.item(),
                                                                                                loss.item()))
        else:
            print('epoch {}, end2end loss {:.6f}'.format(epoch,MSE_loss.item()))

    def moving_average(a, window_size=50):
        return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]


    # test the model


    import matplotlib.pyplot as plt
    plt.switch_backend('TkAgg')
    plt.plot(moving_average(np.array(mse_losses)))
    if calc_sim:
        plt.plot(moving_average(np.array(sim_losses)))
    plt.legend(['MSE','MI'])
    plt.show()


    model.eval()
    total_loss= 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to('cuda')
            outputs = model(data,calc_sim=False)
            MSE_loss = criterion(outputs, data)
            total_loss += MSE_loss.item()

    print(total_loss)