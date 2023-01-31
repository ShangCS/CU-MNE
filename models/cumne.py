import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from evaluate import evaluate
from data import data
from layers import GCN, MLP


class CUMNE(data):
    def __init__(self, args):
        data.__init__(self, args)
        self.args = args
        self.t = args.t
        self.coef_intra = self.args.coef_intra
        self.coef_f = self.args.coef_f
        self.coef_inter = self.args.coef_inter

        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder)

    def training(self):
        features = self.features.to(self.args.device)
        features_pos = self.features_pos.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]
        adj_pos_list = [adj.to(self.args.device) for adj in self.adj_pos_list]

        print("Start training...")
        model = modeler(self.args.ft_size, self.args.hid_units, len(adj_list), self.t).to(self.args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        cnt_wait = 0
        best = 1e9
        model.train()
        for _ in tqdm(range(self.args.nb_epochs)):
            optimizer.zero_grad()
            # corruption function
            idx = np.random.permutation(self.args.nb_nodes)
            features_neg = features[idx, :].to(self.args.device)

            loss_intra, loss_f, loss_inter = model(features, features_pos, features_neg, adj_list, adj_pos_list, self.args.sparse)

            loss = self.coef_intra * loss_intra + self.coef_f * loss_f + self.coef_inter * loss_inter

            # early stop
            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), 'Results/model/{}_temp.pkl'.format(self.args.dataset))
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                print("Early stopped!")
                break

            loss.backward()
            optimizer.step()
        
        # save
        model.load_state_dict(torch.load('Results/model/{}_temp.pkl'.format(self.args.dataset)))

        # evaluation
        print("Evaluating...")
        model.eval()
        embeds = model.embed(features, adj_list, self.args.sparse)
        macro_f1s, micro_f1s, nmi, sim = evaluate(embeds, self.idx_train, self.idx_val, self.idx_test, self.labels, self.args.dataset, )
        
        return macro_f1s, micro_f1s, nmi, sim

class modeler(nn.Module):
    def __init__(self, ft_size, hid_units, n_networks, t):
        super(modeler, self).__init__()
        self.t = t
        self.gcn_list = nn.ModuleList([GCN(ft_size, hid_units) for _ in range(n_networks)])
        self.com_mlp = MLP(hid_units)
        self.uni_mlp = MLP(hid_units)
        
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    def get_intra_loss(self, h, h_pos, h_neg):
        loss = 0
        score_pos = torch.exp(self.cos(h, h_pos)/self.t)
        score_neg = torch.exp(self.cos(h, h_neg)/self.t)
        loss += -torch.sum(torch.log(score_pos/(score_pos+score_neg)))/h.shape[0]

        return loss
    
    def get_inter_loss(self, h_list, h_neg_list):
        loss = 0
        for i, h in enumerate(h_list):
            h_neg = h_neg_list[i]
            h_pos_list = []
            for j in range(len(h_list)):
                if i!=j:
                    h_pos_list.append(h_list[j])
            loss_v = 0
            for p in range(len(h_pos_list)):
                score_pos = torch.exp(self.cos(h_pos_list[p], h)/self.t)
                score_neg = torch.exp(self.cos(h_neg, h)/self.t)
                score = -torch.log(score_pos/(score_pos+score_neg))
                loss_v += torch.sum(score)
            loss += loss_v
        loss /= ((len(h_list)-1)*h_list[0].shape[0])
        
        return loss  
    
    def forward(self, features, features_pos, features_neg, adj_list, adj_pos_list, sparse):
        h_com_list = []
        h_com_neg_list = []
        
        loss_intra = loss_f = loss_inter = 0
        for i, adj in enumerate(adj_list):
            # node embedding
            h_temp = self.gcn_list[i](features, adj, sparse)
            # node common embedding
            h_com = torch.squeeze(self.com_mlp(h_temp))
            h_com_list.append(h_com)
            # node unique embedding
            h_uni = torch.squeeze(self.uni_mlp(h_temp))     

            # node negative embedding
            h_neg_temp = self.gcn_list[i](features_neg, adj, sparse)
            # node common embedding
            h_com_neg = torch.squeeze(self.com_mlp(h_neg_temp))
            h_com_neg_list.append(h_com_neg)
            # node unique embedding
            h_uni_neg = torch.squeeze(self.uni_mlp(h_neg_temp))       
            
            # node positive embedding
            h_pos_temp = self.gcn_list[i](features_pos, adj_pos_list[i], sparse)
            # node common embedding
            h_com_pos = torch.squeeze(self.com_mlp(h_pos_temp))
            # node unique embedding
            h_uni_pos = torch.squeeze(self.uni_mlp(h_pos_temp))
            
            loss_intra += self.get_intra_loss(h_com, h_com_pos, h_com_neg)
            loss_intra += self.get_intra_loss(h_uni, h_uni_pos, h_uni_neg)
            
            loss_f += self.get_intra_loss(h_uni, h_com, h_uni_neg)
        
        # Inter-view loss
        loss_inter = self.get_inter_loss(h_com_list, h_com_neg_list)
        
        return loss_intra, loss_f, loss_inter

    # return embedding
    def embed(self, features, adj_list, sparse):
        h_list = []
        for i, adj in enumerate(adj_list):
            h_temp = self.gcn_list[i](features, adj, sparse)
            h_list.append(torch.squeeze(h_temp))
        h_mean = torch.mean(torch.stack(h_list, 0), 0)

        return h_mean.detach()