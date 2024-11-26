import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from transformers import AutoConfig, BertModel
from transformers.models.bert.modeling_bert import BertLayer
from zmq import device

from .coattention import *
from .layers import *
from utils.metrics import *
from collections import Counter
from transformers import AutoTokenizer, BertModel





class CrossModalMemory(nn.Module):
    def __init__(self, memory_size=200, feature_dim=512):
        super(CrossModalMemory, self).__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        # 初始化记忆矩阵，大小为 (memory_size, feature_dim)
        self.memory = nn.Parameter(torch.randn(memory_size, feature_dim))
        # 训练参数矩阵，用于对齐 multimodal 特征与 memory 向量
        self.W_m = nn.Linear(feature_dim * 2, memory_size)

    def forward(self, visual_feat, text_feat):
        # 融合视觉特征和文本特征
        x_vt = torch.cat([visual_feat, text_feat], dim=-1)
        # 计算 multimodal 特征与 memory 向量的相似性分数
        scores = F.softmax(self.W_m(x_vt), dim=-1)
        # 选择最相关的 memory 向量
        relevant_memory = torch.matmul(scores, self.memory)
        return relevant_memory

    def memory_ranking(self, visual_feat, text_feat):
        # 排序 memory 向量并返回最相关的 top-k 向量
        x_vt = torch.cat([visual_feat, text_feat], dim=-1)
        scores = F.softmax(self.W_m(x_vt), dim=-1)
        ranked_memory_indices = torch.argsort(scores, descending=True)
        return ranked_memory_indices

    def memory_sampling(self, visual_feat, text_feat, k=5):
        # 记忆采样，基于相似性分数从 top-k 中随机采样记忆向量
        ranked_indices = self.memory_ranking(visual_feat, text_feat)
        top_k_indices = ranked_indices[:, :k]
        sampled_memory = self.memory[top_k_indices]
        mx = sampled_memory.mean(dim=1)
        return mx


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, activation=F.relu, dropout=0.0):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # Define the hidden layers
        layers = []
        if num_layers > 1:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Forward pass through all layers
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        # Last layer without activation
        x = self.layers[-1](x)
        return x


class BMRMModel(torch.nn.Module):
    def __init__(self,bert_model,fea_dim,dropout):
        super(BMRMModel, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model).requires_grad_(False)
        # self.bert_uncased = BertModel.from_pretrained('/data/kxz/AAAI/code/bert-base-uncased').requires_grad_(False)

        self.text_dim = 768
        self.comment_dim = 768
        self.img_dim = 4096
        self.video_dim = 4096
        self.num_frames = 83
        self.num_audioframes = 50
        self.num_comments = 23
        self.dim = fea_dim
        self.num_heads = 4

        self.dropout = dropout

        self.attention = Attention(dim=self.dim,heads=4,dropout=dropout)

        self.vggish_layer = torch.hub.load('/data/kxz/FakeSV-main/code/torchvggish/', 'vggish', source = 'local')        
        net_structure = list(self.vggish_layer.children())      
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])

        self.co_attention_ta = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout, d_model=fea_dim,
                                        visual_len=self.num_audioframes, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)
        self.co_attention_tv = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout, d_model=fea_dim,
                                        visual_len=self.num_frames, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)
        self.co_attention_va = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout, d_model=fea_dim,
                                        visual_len=self.video_dim, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)
        self.trm = nn.TransformerEncoderLayer(d_model = self.dim, nhead = 2, batch_first = True)

        # self.trm = nn.TransformerDecoderLayer(d_model=self.dim, nhead=2, batch_first=True)
        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_comment = nn.Sequential(torch.nn.Linear(self.comment_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_video = nn.Sequential(torch.nn.Linear(self.video_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_intro = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim),torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(fea_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))

        self.classifier = nn.Linear(128,2)
        self.classifier = nn.Linear(128, 2)

        self.linner = nn.Sequential(torch.nn.Linear(4096, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_ta = nn.Sequential(torch.nn.Linear(256, 128), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_iv = nn.Sequential(torch.nn.Linear(256, 128), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_memory = nn.Sequential(torch.nn.Linear(256, 128), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.memory = CrossModalMemory(memory_size=200, feature_dim=256)# self.memory_module = MemoryModule()
        self.linear_without = nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ReLU(),nn.Dropout(p=self.dropout))

        self.mlp = MLP(input_dim=128, hidden_dim=64, output_dim=4, num_layers=3, dropout=0.2)
        self.classifier_va = nn.Linear(fea_dim, 2)

        self.encoder1 = nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), nn.Dropout(p=self.dropout))
        self.encoder2 = nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), nn.Dropout(p=self.dropout))
        self.fc_mu1 = nn.Linear(128, 128)
        self.fc_std1 = nn.Linear(128, 128)
        self.fc_mu2 = nn.Linear(128, 128)
        self.fc_std2 = nn.Linear(128, 128)
        self.linear21 = nn.Linear(128 * 2, 128)

        self.decoder1 = nn.Linear(128, 4)
        self.decoder2 = nn.Linear(128, 2)
        self.BCEWithLogitsLoss=nn.BCEWithLogitsLoss()

    def encode1(self, x):
        x = x.cuda()
        x = self.encoder1(x)
        return self.fc_mu1(x), F.softplus(self.fc_std1(x) - 5, beta=1)

    def encode2(self, x):
        x = x.cuda()
        x = self.encoder2(x)
        return self.fc_mu2(x), F.softplus(self.fc_std2(x) - 5, beta=1)

    def reparameterise(self, mu, std):
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode1(self, x):
        x = x.cuda()
        x = self.decoder1(x)
        return F.softmax(x, dim=1)

    def decode2(self, x):
        x = x.cuda()
        x = self.decoder2(x)
        return F.softmax(x, dim=1)
    
    def forward(self,  **kwargs):

        ### User Intro ###
        # intro_inputid = kwargs['intro_inputid']
        # intro_mask = kwargs['intro_mask']
        # fea_intro = self.bert(intro_inputid,attention_mask=intro_mask)[1]
        # fea_intro = self.linear_intro(fea_intro) 

        # ### Title ###
        title_inputid = kwargs['title_inputid']#(batch,512)
        title_mask=kwargs['title_mask']#(batch,512)
        fea_text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']
        # print(title_inputid.shape)
        fea_text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']  # (batch, sequence, 768)
        fea_text=self.bert(title_inputid,attention_mask=title_mask)['last_hidden_state']#(batch,sequence,768)
        fea_text=self.linear_text(fea_text) 
        # print(fea_text.shape)

        # ### Audio Frames ###
        audioframes=kwargs['audioframes']#(batch,36,12288)
        audioframes_masks = kwargs['audioframes_masks']
        fea_audio = self.vggish_modified(audioframes) #(batch, frames, 128)
        fea_audio = self.linear_audio(fea_audio) 
        # fea_audio, fea_text = self.co_attention_ta(v=fea_audio, s=fea_text, v_len=fea_audio.shape[1], s_len=fea_text.shape[1])
        

        # ### Image Frames ###
        frames=kwargs['frames']#(batch,30,4096)
        frames_masks = kwargs['frames_masks']
        fea_img = self.linear_img(frames) 
        fea_img, fea_text = self.co_attention_tv(v=fea_img, s=fea_text, v_len=fea_img.shape[1], s_len=fea_text.shape[1])
        fea_img = torch.mean(fea_img, -2)

        fea_text = torch.mean(fea_text, -2)
        
        # ### C3D ###
        c3d = kwargs['c3d'] # (batch, 36, 4096)
        c3d_masks = kwargs['c3d_masks']
        fea_video = self.linear_video(c3d) #(batch, frames, 128)
        fea_video = torch.mean(fea_video, -2)
        fea_audio = torch.mean(fea_audio, -2)
        fea_ta = torch.cat((fea_text, fea_audio), 1)

        fea_ta = self.linear_without(fea_audio)


        # ### Comment ###
        # comments_inputid = kwargs['comments_inputid']#(batch,20,250)
        # comments_mask=kwargs['comments_mask']#(batch,20,250)

        # comments_like=kwargs['comments_like']
        # comments_feature=[]
        # for i in range(comments_inputid.shape[0]):
        #     bert_fea=self.bert(comments_inputid[i], attention_mask=comments_mask[i])[1]
        #     comments_feature.append(bert_fea)
        # comments_feature=torch.stack(comments_feature) #(batch,seq,fea_dim)

        # fea_comments =[]
        # for v in range(comments_like.shape[0]): 
        #     comments_weight=torch.stack([torch.true_divide((i+1),(comments_like[v].shape[0]+comments_like[v].sum())) for i in comments_like[v]])
        #     comments_fea_reweight = torch.sum(comments_feature[v]*(comments_weight.reshape(comments_weight.shape[0],1)),dim=0)
        #     fea_comments.append(comments_fea_reweight)
        # fea_comments = torch.stack(fea_comments)
        # fea_comments = self.linear_comment(fea_comments)#(batch,fea_dim)

        ##Emotion##
        # Emotion_inputid = kwargs['Emotion_inputid']
        # Emotion_mask = kwargs['Emotion_mask']

        # fea_Emotion = self.bert_uncased(Emotion_inputid, attention_mask=Emotion_mask)['last_hidden_state']
        # fea_Emotion = self.linear_Emotion(fea_Emotion) 
        # fea_Emotion = torch.mean(fea_Emotion, -2)

        # ##Theme
        # Theme_inputid = kwargs['Theme_inputid']
        # Theme_mask = kwargs['Theme_mask']

        # fea_Theme = self.bert_uncased(Theme_inputid, attention_mask=Theme_mask)['last_hidden_state']
        # fea_Theme = self.linear_Theme(fea_Theme) 
        # fea_Theme = torch.mean(fea_Theme, -2)

        # ##Sensitivity
        # Sensitivity_inputid = kwargs['Sensitivity_inputid']
        # Sensitivity_mask = kwargs['Sensitivity_mask']
        
        # fea_Sensitivity = self.bert_uncased(Sensitivity_inputid, attention_mask=Sensitivity_mask)['last_hidden_state']
        # fea_Sensitivity = self.linear_Sensitivity(fea_Sensitivity) 
        # fea_Sensitivity = torch.mean(fea_Sensitivity, -2)

        # ##Rumor
        # Rumor_inputid = kwargs['Rumor_inputid']
        # Rumor_mask = kwargs['Rumor_mask']

        # fea_Rumor = self.bert_uncased(Rumor_inputid, attention_mask=Rumor_mask)['last_hidden_state']
        # fea_Rumor = self.linear_Rumor(fea_Rumor) 
        # fea_Rumor = torch.mean(fea_Rumor, -2)

        # ##Match
        # Match_inputid = kwargs['Match_inputid']
        # Match_mask = kwargs['Match_mask']

        # fea_Match = self.bert_uncased(Match_inputid, attention_mask=Match_mask)['last_hidden_state']
        # fea_Match = self.linear_Match(fea_Match) 
        # fea_Match = torch.mean(fea_Match, -2)




        # embs = kwargs['embs']
        # embs = torch.mean(embs, -2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linner.to(device)
        # embs = embs.to(device)
        # fea_emb = self.linner(embs)
        # fea_emb, fea_text = self.co_attention_te(v=fea_emb, s=fea_text, v_len=fea_emb.shape[0], s_len=fea_text.shape[1])

        fea_iv = torch.cat((fea_img,  fea_video), 1)


        # fea_iv=self.linear_without(fea_img)
        

        # 前向传播，获得融合后的记忆向量
        memory_output = self.memory(fea_iv, fea_ta)
    
        # 记忆采样，获得对比信息融合向量
        memory_output = self.memory.memory_sampling(fea_iv, fea_ta, k=5)


        fea_ta = self.linear_ta(fea_ta).unsqueeze(1)
        fea_iv = self.linear_iv(fea_iv).unsqueeze(1)
        fea_va = torch.cat((fea_ta, fea_iv ), 1)  # (batchsize,2,128)
        fea_va1 = self.trm(fea_va)
        fea_va2 = torch.mean(fea_va1, -2)
        va_mu, va_std = self.encode1(fea_va2)
        va_y = self.reparameterise(va_mu, va_std)
        va_y = self.decode1(va_y)
        memory_output = self.linear_memory(memory_output).unsqueeze(1)




        fea = torch.cat((fea_iv, fea_ta,memory_output), 1)
        # fea = torch.cat((fea_iv, fea_ta), 1)





        # fea=torch.cat((fea_text,fea_audio, fea_video,fea_img),1) # (bs, 6, 128)
        fea = self.trm(fea)
        fea = torch.mean(fea, -2)




        output = self.mlp(fea)
        
        # output = self.classifier(fea)

        return output, fea, va_y, va_mu, va_std
