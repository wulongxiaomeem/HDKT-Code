import math
import logging
import torch
import torch.nn as nn
import dhg
from sklearn import metrics
import tqdm
import numpy as np
import random
import os
device = torch.device('cuda:3' if torch.cuda.is_available else 'cpu' )
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from sklearn.metrics import mean_squared_error
from math import sqrt
import copy
import pickle

def seed_everything(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(111)

batch_size = 32
n_s = 10000             #学生的数量
max_seqlen =500         #学生序列的最长长度

num_dict = {
    'n_e' :  11964,            #题目的数量
    'n_c'  : 188,              #知识点的数量
    'n_dl' : 100,              #难度的数量
    'n_it' : 7,                #间隔时间
    'n_ep' : 100,              #试题流行度
    'n_h_attempt' : 10,        #总尝试次数
   'n_c_attempt' : 10,        #连续尝试次数
    'n_as' : 7               #回答速度
    
}

embedding_dict = {
    'd_e' : 128,             
    'd_c' : 128,            
    'd_dl' : 128,              
    'd_a' : 128,               
    'd_it' : 128,              
    'd_ep' : 128,             
    'd_attempt' : 128,          
    'd_x' : 128,               
    'd_as' : 128,               
    'd_h' : 128
}

dropout = 0

Q_matrix_c = np.loadtxt('matrix/Q_matrix_idx.txt')
Q_matrix_cd = np.loadtxt('matrix/Q_matrix_kcdifficulty.txt')

'''
loading edges
'''
with open('edge/exercise-concept_edges.pkl', 'rb') as file:
    ec_edges = pickle.load(file)
with open('edge/exercise-difficulty_edges.pkl', 'rb') as file:
    ed_edges = pickle.load(file)
with open('edge/exercise-popularity_edges.pkl', 'rb') as file:
    ep_edges = pickle.load(file)

'''
create hypergraph
'''
hg = dhg.Hypergraph(num_dict['n_e'] + 1, ec_edges,device = device)
hg.add_hyperedges(ed_edges, group_name = "ed")# (optional parameter)e_weight: A list of weights for hyperedges
hg.add_hyperedges(ep_edges, group_name = "ep")# (optional parameter)e_weight: A list of weights for hyperedges

def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0-target)*np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0

def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)

def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)

class DATA(object):
    def __init__(self, max_seqlen, separate_char):
        self.separate_char = separate_char
        self.max_seqlen = max_seqlen
    '''
    sequence length
    student id, 
    exercise id, 
    exercise difficulty id, 
    exercise popularity id, 
    answer, 
    answer speed,
    historical attempt times,
    historical attempt times, 
    iterval time
    '''
    
    def load_data(self, path):
        f_data  = open(path, 'r')
        
        e_data  = []
        ed_data  = []
        ep_data = []
        a_data = []
        as_data  = []
        ha_data = []
        ca_data = []
        it_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 10 !=0:
                line_data = line.split(self.separate_char)
                if len(line_data[len(line_data) - 1]) == 0: 
                    line_data = line_data[:-1]
            
            if lineID % 10 == 2:
                E = line_data
            elif lineID % 10 ==3:
                ED = line_data
            elif lineID % 10 ==4:
                EP = line_data
            elif lineID % 10 ==5:
                A = line_data
            elif lineID % 10 ==6:
                AS  = line_data
            elif lineID % 10 ==7:
                HA  = line_data
            elif lineID % 10 ==8:
                CA  = line_data
            elif lineID % 10 ==9:
                IT  = line_data
                
                E  = list(map(int, E))
                ED  = list(map(int, ED))
                EP = list(map(int, EP))
                A = list(map(int, A))
                AS  = list(map(int, AS))
                HA = list(map(int, HA))
                CA = list(map(int, CA))
                IT = list(map(int, IT))

                
                e_data.append(E)
                ed_data.append(ED)
                ep_data.append(EP)
                a_data.append(A)
                as_data.append(AS)
                ha_data.append(HA)
                ca_data.append(CA)
                it_data.append(IT)
                
        f_data.close()
        
        E_dataArray = np.zeros((len(e_data), self.max_seqlen)) 
        for j in range(len(e_data)):
            dat = e_data[j]
            E_dataArray[j,:len(dat)] = dat
        
        ED_dataArray = np.zeros((len(ed_data), self.max_seqlen)) 
        for j in range(len(ed_data)):
            dat = ed_data[j]
            ED_dataArray[j,:len(dat)] = dat
            
        EP_dataArray = np.zeros((len(ep_data), self.max_seqlen))  
        for j in range(len(ep_data)):
            dat = ep_data[j]
            EP_dataArray[j,:len(dat)] = dat
        
        A_dataArray = np.zeros((len(a_data), self.max_seqlen))
        for j in range(len(a_data)):
            dat = a_data[j]
            A_dataArray[j,:len(dat)] = dat
            
        AS_dataArray = np.zeros((len(as_data), self.max_seqlen)) 
        for j in range(len(as_data)):
            dat = as_data[j]
            AS_dataArray[j,:len(dat)] = dat
        
        HA_dataArray = np.zeros((len(ha_data), self.max_seqlen))
        for j in range(len(ha_data)):
            dat = ha_data[j]
            HA_dataArray[j,:len(dat)] = dat
            
        CA_dataArray = np.zeros((len(ca_data), self.max_seqlen)) 
        for j in range(len(ca_data)):
            dat = ca_data[j]
            CA_dataArray[j,:len(dat)] = dat
        IT_dataArray = np.zeros((len(it_data), self.max_seqlen))
        for j in range(len(it_data)):
            dat = it_data[j]
            IT_dataArray[j,:len(dat)] = dat
        return E_dataArray, ED_dataArray, EP_dataArray, A_dataArray, AS_dataArray, HA_dataArray,CA_dataArray,IT_dataArray

class HyKTNet(nn.Module):
    def __init__(self, num_dict, embedding_dict, hg, Q_matrix_c, Q_matrix_cd, dropout = 0.2):
        super(HyKTNet, self).__init__()
        
        self.n_e = num_dict['n_e']  
        self.n_c = num_dict['n_c']  
        self.n_dl = num_dict['n_dl']
        self.n_it = num_dict['n_it']
        self.n_ep = num_dict['n_ep']
        self.n_ha = num_dict['n_h_attempt']
        self.n_ca = num_dict['n_c_attempt']
        self.n_as = num_dict['n_as']

        self.d_e = embedding_dict['d_e'] 
        self.d_c = embedding_dict['d_c'] 
        self.d_dl = embedding_dict['d_dl']  
        self.d_a = embedding_dict['d_a']  
        self.d_it = embedding_dict['d_it'] 
        self.d_ep = embedding_dict['d_ep'] 
        self.d_attempt = embedding_dict['d_attempt'] 
        self.d_x = embedding_dict['d_x'] 
        self.d_as = embedding_dict['d_as'] 
        self.d_h = embedding_dict['d_h']
 
        self.ec_matrix   = Q_matrix_c
        self.ecd_matrix  = Q_matrix_cd
        self.hg = hg
        
        self.e_embed  = nn.Embedding(self.n_e+1, self.d_e)         
        torch.nn.init.xavier_uniform_(self.e_embed.weight)
        self.c_embed  = nn.Embedding(self.n_c+1, self.d_c, padding_idx=0)         
        torch.nn.init.xavier_uniform_(self.c_embed.weight) 
        self.ed_embed = nn.Embedding(self.n_dl+1, self.d_dl)          
        torch.nn.init.xavier_uniform_(self.ed_embed.weight)
        self.cd_embed = nn.Embedding(self.n_dl+1, self.d_dl, padding_idx=0)          
        torch.nn.init.xavier_uniform_(self.cd_embed.weight)    
        self.a_embed  = nn.Embedding(2, self.d_a)             
        torch.nn.init.xavier_uniform_(self.a_embed.weight)
        self.it_embed = nn.Embedding(self.n_it+1, self.d_it)         
        torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.ep_embed = nn.Embedding(self.n_ep+1, self.d_ep)        
        torch.nn.init.xavier_uniform_(self.ep_embed.weight)
        self.ha_embed = nn.Embedding(self.n_ha+1, self.d_attempt)        
        torch.nn.init.xavier_uniform_(self.ha_embed.weight)
        self.ca_embed = nn.Embedding(self.n_ca+1, self.d_attempt)       
        torch.nn.init.xavier_uniform_(self.ca_embed.weight)
        self.as_embed = nn.Embedding(self.n_as+1, self.d_as)       
        torch.nn.init.xavier_uniform_(self.as_embed.weight)
        
        self.HgnnpConv1 = dhg.nn.HGNNPConv(self.d_e, self.d_e, bias = True, use_bn = True, drop_rate = 0.8, is_last = False)
        self.HgnnpConv2 = dhg.nn.HGNNPConv(self.d_e, self.d_e, bias = True, use_bn = True, drop_rate = 0.8, is_last = True)
 
        self.linear_i = nn.Linear(self.d_e + self.d_dl + self.d_ep + self.d_c + self.d_dl, self.d_x)       
        torch.nn.init.xavier_uniform_(self.linear_i.weight)

        self.linear_predicting_s = nn.Linear(self.d_attempt + self.d_attempt + self.d_as + self.d_it, 1)
        torch.nn.init.kaiming_uniform_(self.linear_predicting_s.weight) 
        
        self.linear_predicting_e = nn.Linear(self.d_x, 1)
        torch.nn.init.kaiming_uniform_(self.linear_predicting_e.weight) 
        
        self.linear_forgetting = nn.Linear(self.d_h + self.d_it, self.d_h)
        torch.nn.init.xavier_uniform_(self.linear_forgetting.weight) 
        
        self.linear_prepredicting = nn.Linear(self.d_x + self.d_h + self.d_attempt + self.d_attempt, self.d_x)
        torch.nn.init.kaiming_uniform_(self.linear_prepredicting.weight) 
        
        self.linear_predicting = nn.Linear(self.d_x, 1)
        torch.nn.init.xavier_uniform_(self.linear_predicting.weight) 
        
        self.linear_learning = nn.Linear(self.d_x + self.d_h + self.d_a + self.d_attempt + self.d_attempt + self.d_as, self.d_h)
        torch.nn.init.xavier_uniform_(self.linear_learning.weight)
        
        self.linear_learning_gate = nn.Linear(self.d_x + self.d_h + self.d_a + self.d_attempt + self.d_attempt + self.d_as, self.d_h)
        torch.nn.init.xavier_uniform_(self.linear_learning_gate.weight)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.soft = nn.Softmax(dim=2)

    def forward(self, e_data, ed_data, ep_data, a_data, as_data, ha_data, ca_data, it_data,deliver_h=None, deliver_h_list=[], train_epoch = False):
        
        batch_size, seq_len = e_data.size(0), e_data.size(1)
        exercises = torch.arange(self.n_e+1).to(device)
        concepts = torch.arange(self.n_c+1).to(device)
    
        ed_embed_data = self.ed_embed(ed_data)#(batch_size, seq_len, d_l) 
        a_embed_data  = self.a_embed(a_data)  #(batch_size, seq_len, d_a) 
        it_embed_data = self.it_embed(it_data) #(batch_size, seq_len, d_a) 
        ep_embed_data = self.ep_embed(ep_data)      #(batch_size, seq_len, d_ep) 
        ha_embed_data = self.ha_embed(ha_data)      #(batch_size, seq_len, d_ha) 
        ca_embed_data = self.ca_embed(ca_data)      #(batch_size, seq_len, d_ca) 
        as_embed_data = self.as_embed(as_data)      #(batch_size, seq_len, d_as) 
        
        exercises_embedding = self.e_embed(exercises) #(n_e + 1, 1, d_e)
        concepts_embedding = self.c_embed(concepts)  #(n_c + 1, d_c)
        e_embed_conv1 = self.HgnnpConv1(exercises_embedding, self.hg)
        e_embed_conv2 = self.HgnnpConv2(e_embed_conv1, self.hg)
        e_embed_conv2 = e_embed_conv2 + exercises_embedding
        c2mean = (self.ec_matrix != 0).sum(1).view(-1,1) #to_mean[0] = 1
        embed_ec_matrix = self.c_embed(self.ec_matrix)
        embed_ecd_matrix = self.cd_embed(self.ecd_matrix)
        scores = (e_embed_conv2.reshape(-1, 1, self.d_e).repeat(1, self.n_c+1, 1) * embed_ec_matrix).sum(2)/(self.d_e ** 1/2)
        scores = scores.masked_fill(self.ec_matrix==0, -1e23)
        soft_scores = nn.Softmax()(scores)
        c_embed_data  = ((embed_ec_matrix*((soft_scores.unsqueeze(2)).repeat(1,1,self.d_c))).sum(1))[e_data]
        cd_embed_data  = ((embed_ecd_matrix*((soft_scores.unsqueeze(2)).repeat(1,1,self.d_dl))).sum(1))[e_data]
        retain_h = torch.zeros((batch_size, self.d_h)).to(device)
        if  train_epoch: 
            h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, self.d_h)).to(device)
        else: 
            h_pre = deliver_h
            
        e_embed_data = e_embed_conv2[e_data]
        i_embed_data = self.linear_i(torch.cat((e_embed_data, ed_embed_data, ep_embed_data, c_embed_data, cd_embed_data),2))

        search_idx =[i for i in range(batch_size)]
        copy_idx =[i for i in range(batch_size)]
        
        pred_s = torch.zeros(batch_size, seq_len).to(device)  
        pred_main = torch.zeros(batch_size, seq_len).to(device)    
        
        one = torch.ones(batch_size, 1).to(device)
        min_learing = torch.tensor(0.4).to(device)
        
        one_f = torch.ones(1, self.d_h).to(device)
        for t in range(0, seq_len):
            i = i_embed_data[:, t]#(batch_size, d_x)
            a = a_embed_data[:, t]
            it = it_embed_data[:, t]
            ep = ep_embed_data[:, t]
            ha = ha_embed_data[:, t]
            ca = ca_embed_data[:, t]
            AS = as_embed_data[:, t]
            eid = e_data[:, t]  #(batch_size)
            aid = a_data[:, t].view(-1, 1)  #(batch_size, 1)
            itid = it_data[:, t].view(-1, 1)

            '''
            s_part
            '''
            s_pred = self.sig(self.linear_predicting_s(torch.cat((ha, ca, AS, it),1)))
            pred_s[:, t] = s_pred.view(-1)

                              
            '''
            main_part
            '''

            fg = []
            for b in range(batch_size):
                if(itid[b] > 1):
            
                    forgetting_gate = self.sig(self.linear_forgetting(torch.cat((
                    h_pre[b].unsqueeze(0),
                    it[b].unsqueeze(0)
                       ), 1)))
                    fg.append(forgetting_gate)
                else:
                    fg.append(one_f)
            h_pre_f = torch.cat(fg, dim=0) * h_pre
            
            main_prepred = self.linear_prepredicting(torch.cat((i, h_pre_f, ha, ca),1))
            main_prepred = self.dropout(self.relu(main_prepred))
            main_pred = self.sig(self.linear_predicting(main_prepred))
            pred_main[:, t] = main_pred.view(-1)

            a_tilde = aid * (torch.max(main_pred -  s_pred, min_learing) * a) + (one - aid) * a
            learning_gain = self.linear_learning(torch.cat((i, h_pre_f, a_tilde, AS, ha, ca),1))  #(batch_size, d_x)
            learning_gain = self.sig(learning_gain)
            learning_gain_gate = self.linear_learning_gate(torch.cat((i, h_pre_f, a_tilde, AS, ha, ca),1))  #(batch_size, d_x)
            learning_gain_gate = self.sig(learning_gain_gate) 
            
            learning_gain = learning_gain * (1 - learning_gain_gate)
            h = learning_gain + h_pre_f * learning_gain_gate  #(batch_size , self.n_c + 1, d_x)

            for i in search_idx :
                if  t==seq_len-1 or e_data[i,t+1]==0:                    
                    retain_h[i]=h[i]
                    copy_idx.remove(i)
            search_idx = copy.deepcopy(copy_idx)

            h_pre = h 
            
            if train_epoch and t==seq_len-1:
                deliver_h_list.append(retain_h)
        return pred_s, pred_main

def train_one_epoch(net, retain_h, optimizer, criterion, batch_size, e_data, ed_data, ep_data, a_data, as_data, ha_data, ca_data, it_data):
    net.train()
    n = int(math.ceil(len(e_data) / batch_size ))
    shuffled_ind = np.arange(e_data.shape[0])
    np.random.shuffle(shuffled_ind)
    
    e_data = e_data[shuffled_ind]
    ed_data = ed_data[shuffled_ind]
    a_data = a_data[shuffled_ind]
    it_data = it_data[shuffled_ind]
    ep_data = ep_data[shuffled_ind]
    ha_data = ha_data[shuffled_ind]
    ca_data = ca_data[shuffled_ind]
    as_data = as_data[shuffled_ind]
    
    pred_list = []
    target_list = []
    
    for idx in tqdm.tqdm(range(n), 'Training'):
        optimizer.zero_grad()
        e_one_seq = e_data[idx *batch_size :(idx+1) * batch_size, :]
        ed_one_seq = ed_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq  = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq  = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        ep_one_seq  = ep_data[idx * batch_size: (idx + 1) * batch_size, :]
        ha_one_seq  = ha_data[idx * batch_size: (idx + 1) * batch_size, :]
        ca_one_seq  = ca_data[idx * batch_size: (idx + 1) * batch_size, :]
        as_one_seq  = as_data[idx * batch_size: (idx + 1) * batch_size, :]
        
        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_ed = torch.from_numpy(ed_one_seq).long().to(device)
        input_a = torch.from_numpy(a_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        input_ep = torch.from_numpy(ep_one_seq).long().to(device)
        input_ha= torch.from_numpy(ha_one_seq).long().to(device)
        input_ca= torch.from_numpy(ca_one_seq).long().to(device)
        input_as = torch.from_numpy(as_one_seq).long().to(device)
        
        target = torch.from_numpy(a_one_seq).float().to(device)

        pred_s, pred_main = net(input_e, input_ed, input_ep, input_a, input_as, input_ha, input_ca, input_it, deliver_h=None, train_epoch = True, deliver_h_list=retain_h)

        mask = input_e[:, 1:]>0
        masked_truth = target[:, 1:][mask]
        masked_pred_s = pred_s[:, 1:][mask]
        masked_pred_main = pred_main[:, 1:][mask]
        
        loss = 0.01 * criterion(masked_pred_s, masked_truth).sum() +  criterion(masked_pred_main, masked_truth).sum()
        loss.backward()
        optimizer.step()

        masked_pred_main = masked_pred_main.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()
        
        pred_list.append(masked_pred_main)
        target_list.append(masked_truth)
    
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    
    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy, shuffled_ind

def test_one_epoch(net, batch_size, retain_h, shuffled_ind, e_data, ed_data, ep_data, a_data, as_data, ha_data, ca_data, it_data):
    net.eval()
    n = int(math.ceil(len(e_data) / batch_size ))

    pred_list = []
    target_list = []
    
    e_data = e_data[shuffled_ind]
    ed_data = ed_data[shuffled_ind]
    a_data = a_data[shuffled_ind]
    it_data = it_data[shuffled_ind]
    ep_data = ep_data[shuffled_ind]
    ha_data = ha_data[shuffled_ind]
    ca_data = ca_data[shuffled_ind]
    as_data = as_data[shuffled_ind]

    for idx in tqdm.tqdm(range(n), 'Testing'):
        e_one_seq = e_data[idx *batch_size :(idx+1) * batch_size, :]
        ed_one_seq = ed_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq  = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq  = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        ep_one_seq  = ep_data[idx * batch_size: (idx + 1) * batch_size, :]
        ha_one_seq  = ha_data[idx * batch_size: (idx + 1) * batch_size, :]
        ca_one_seq  = ca_data[idx * batch_size: (idx + 1) * batch_size, :]
        as_one_seq  = as_data[idx * batch_size: (idx + 1) * batch_size, :]
        
        retain_h_one_seq = retain_h[idx * batch_size: (idx + 1) * batch_size, :]

        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_ed = torch.from_numpy(ed_one_seq).long().to(device)
        input_a = torch.from_numpy(a_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        input_ep = torch.from_numpy(ep_one_seq).long().to(device)
        input_ha = torch.from_numpy(ha_one_seq).long().to(device)
        input_ca = torch.from_numpy(ca_one_seq).long().to(device)
        input_as = torch.from_numpy(as_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)

        with torch.no_grad():
            pred_s, pred_main = net(input_e, input_ed, input_ep, input_a, input_as, input_ha, input_ca, input_it, deliver_h = retain_h_one_seq, train_epoch = False, deliver_h_list=[])

            mask = input_e[:, 1:] > 0
            masked_truth = target[:, 1:][mask].detach().cpu().numpy()
            masked_pred_s = pred_s[:, 1:][mask].detach().cpu().numpy()
            masked_pred_main = pred_main[:, 1:][mask].detach().cpu().numpy()

            pred_list.append(masked_pred_main)
            target_list.append(masked_truth)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    MSE = mean_squared_error(all_target, all_pred)**0.5
    MAE = metrics.mean_absolute_error(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    print("MSE: %.6f", MSE)
    print("MAE: %.6f", MAE)

    return loss, auc, accuracy

class HyKT(nn.Module):
    def __init__(self, num_dict, embedding_dict, hg, Q_matrix_c, Q_matrix_cd, batch_size, dropout = 0.2):
        super(HyKT, self).__init__()
        Q_matrix_c = torch.from_numpy(Q_matrix_c).long().to(device)
        Q_matrix_cd = torch.from_numpy(Q_matrix_cd).long().to(device)
        
        self.hykt_net = HyKTNet(num_dict, embedding_dict, hg, Q_matrix_c, Q_matrix_cd, dropout = dropout).to(device)
        self.batch_size = batch_size
    
    def train(self, train_data, test_data=None,*,epoch:int, lr=0.002, lr_decay_step=15, lr_decay_rate = 0.1) ->...:
        train_loss_list = []
        train_auc_list  = []
        test_loss_list  = []
        test_auc_list   = []

        criterion = nn.BCELoss(reduction='none')
        best_train_auc, best_test_auc = .0, .0
        
        optimizer = torch.optim.AdamW(self.hykt_net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)

        for idx in range(epoch):
            retain_h = []
            train_loss, train_auc, train_accuracy, shuffled_ind = train_one_epoch(self.hykt_net, retain_h, optimizer, criterion, self.batch_size, *train_data)

            train_loss_list.append(train_loss)
            train_auc_list.append(train_auc)
            
            
            print("Train  LogisticLoss: %.6f" %  train_loss)
            
            if train_auc > best_train_auc:
                best_train_auc = train_auc
                
            scheduler.step()
            
            for i in range(1,len(retain_h)):
                retain_h[0] = torch.cat((retain_h[0],retain_h[i]),dim=0)

            deliver_h = retain_h[0].to(device)

            if test_data is not None:
                test_loss, test_auc, test_accuracy= self.eval( shuffled_ind, deliver_h, test_data)

                test_loss_list.append(test_loss)
                test_auc_list.append(test_auc)
                
                print("[Epoch %d],Train auc: %.6f, Test auc: %.6f, Test acc: %.6f" % (idx ,train_auc, test_auc, test_accuracy))
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    self.save("hykt.params")  
        return best_train_auc, best_test_auc, train_loss_list, train_auc_list, test_loss_list, test_auc_list
    def eval(self, shuffled_ind, deliver_h, test_data) -> ...:
        self.hykt_net.eval()
        return test_one_epoch(self.hykt_net, self.batch_size, deliver_h, shuffled_ind, *test_data)
    def save(self, filepath) -> ...:
        torch.save(self.hykt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)
    
    def load(self, filepath) -> ...:
        self.hykt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)

dat_train = DATA(max_seqlen=math.ceil(max_seqlen*0.6), separate_char=',')
dat_test  = DATA(max_seqlen=math.ceil(max_seqlen*0.2), separate_char=',')
dat_total_train = DATA(max_seqlen=math.ceil(max_seqlen*0.8), separate_char=',')
logging.getLogger().setLevel(logging.INFO)

train_data = dat_total_train.load_data('dataset/total_train_filtered.txt')
test_data = dat_test.load_data('dataset/test_filtered.txt')

hykt = HyKT(num_dict, embedding_dict, hg, Q_matrix_c = Q_matrix_c, Q_matrix_cd = Q_matrix_cd, batch_size = batch_size, dropout = dropout)

best_train_auc, best_valid_auc, train_loss_list, train_auc_list, test_loss_list, test_auc_list = hykt.train(train_data, test_data, epoch=20, lr=0.002, lr_decay_step=10)
print(' best train auc %f, best test auc %f' % ( best_train_auc , best_valid_auc ))

best_list = [best_train_auc, best_valid_auc]

train_and_valid_best_auc_tuple = np.array(best_list)
np.save('fold_train_best_auc_list.npy',train_and_valid_best_auc_tuple)

fold_train_loss_list_np=np.array(train_loss_list)
np.save('fold_train_loss_list_np.npy',fold_train_loss_list_np)

fold_train_auc_list_np=np.array(train_auc_list)
np.save('fold_train_auc_list_np.npy',fold_train_auc_list_np)

fold_valid_loss_list_np=np.array(test_loss_list)
np.save('fold_valid_loss_list_np.npy',fold_valid_loss_list_np)

fold_valid_auc_list_np=np.array(test_auc_list)
np.save('fold_valid_auc_list_np.npy',fold_valid_auc_list_np)