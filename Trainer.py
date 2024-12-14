import copy
import json
import os
import time
from tkinter import E

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from transformers import BertModel
from utils.metrics import *
from zmq import device

from .coattention import *
from .layers import *

alph=0.1
beta=1e-3

class Trainer():
    def __init__(self,
                model, 
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer, 
                 epoch_stop,
                 epoches,
                 mode,
                 model_name, 
                 event_num,
                 save_threshold = 0.0, 
                 start_epoch = 0,
                 ):
        
        self.model = model
        self.device = device
        self.mode = mode
        self.model_name = model_name
        self.event_num = event_num

        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path= save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.BCEWithLogitsLoss=nn.BCEWithLogitsLoss()
    
        self.criterion = nn.CrossEntropyLoss()

    def klloss(self, outputs, label, mu, std):
        # loss_fct = nn.L1Loss()

        # _, preds = torch.max(outputs, 1)
        # preds=outputs.float()
        # preds=preds.reshape(1,128)
        # print(preds.shape)
        # print(preds.view(-1).shape)
        # _, preds = torch.max(outputs, 1)

        # CE = loss_fct(outputs.view(-1), label.view(-1))
        CE = self.criterion(outputs, label)
        KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2 * std.log() - 1)
        # print("CE:",CE)
        # print("KL:",KL)
        return (beta * KL + CE)
    def fiklloss(self, outputs, output_va, va_mu, va_std, label):

        va_loss = self.klloss(output_va, label, va_mu, va_std)

        final_loss = self.criterion(outputs, label)


        return va_loss * alph + final_loss

    def train(self):

        since = time.time()

        self.model.cuda()

        best_model_wts_test = copy.deepcopy(self.model.state_dict())
        best_acc_test = 0.0
        best_epoch_test = 0
        is_earlystop = False

        if self.mode == "eann":
            best_acc_test_event = 0.0
            best_epoch_test_event = 0

        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch+1, self.start_epoch+self.num_epochs))
            print('-' * 50)

            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
            
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()  
                else:
                    self.model.eval()   
                print('-' * 10)
                print (phase.upper())
                print('-' * 10)

                running_loss_fnd = 0.0
                running_loss = 0.0 
                tpred = []
                tlabel = []
                tmultilabel=[]

                if self.mode == "eann":
                    running_loss_event = 0.0
                    tpred_event = []
                    tlabel_event = []

                for batch in tqdm(self.dataloaders[phase]):
                    batch_data=batch
                    for k,v in batch_data.items():
                        batch_data[k]=v.cuda()
                    label = batch_data['label']
                    multilabel=batch_data['multilabel'].long()
                    if self.mode == "eann":
                        label_event = batch_data['label_event']

                
                    with torch.set_grad_enabled(phase == 'train'):
                        if self.mode == "eann":
                            outputs, outputs_event,fea = self.model(**batch_data)
                            loss_fnd = self.criterion(outputs, label)
                            loss_event = self.criterion(outputs_event, label_event)
                            loss = loss_fnd + loss_event
                            _, preds = torch.max(outputs, 1)
                            _, preds_event = torch.max(outputs_event, 1)
                        else:
                            outputs,fea,output_va, va_mu, va_std = self.model(**batch_data)
                            _, preds = torch.max(outputs, 1)
                            # print(outputs.shape)
                            # print(multilabel.shape)
                            # print(output_va.shape)
                            # print(va_mu.shape)
                            # print(va_std.shape)

                            loss = self.fiklloss(outputs, output_va, va_mu, va_std, multilabel)
                            # loss = self.fiklloss(outputs, output_va, va_mu, va_std, label)
                            # loss = self.criterion(outputs, multilabel)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    #二分类
                    # tlabel.extend(label.detach().cpu().numpy().tolist())
                    # tpred.extend(preds.detach().cpu().numpy().tolist())
                    # running_loss += loss.item() * label.size(0)
                    #四分类
                    tmultilabel.extend(multilabel.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * multilabel.size(0)

                    if self.mode == "eann":
                        tlabel_event.extend(label_event.detach().cpu().numpy().tolist())
                        tpred_event.extend(preds_event.detach().cpu().numpy().tolist())
                        running_loss_event += loss_event.item() * label_event.size(0)
                        running_loss_fnd += loss_fnd.item() * label.size(0)
                    
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                print('Loss: {:.4f} '.format(epoch_loss))

                # 二分类
                # results = metrics(tlabel, tpred)
                # print (results)
                # self.writer.add_scalar('Loss/'+phase, epoch_loss, epoch+1)
                # self.writer.add_scalar('Acc/'+phase, results['acc'], epoch+1)
                # self.writer.add_scalar('F1/'+phase, results['f1'], epoch+1)

                # 四分类
                results = evaluation4class(tmultilabel, tpred)
                print (results)
                self.writer.add_scalar('Loss/'+phase, epoch_loss, epoch+1)
                self.writer.add_scalar('Acc_all/'+phase, results['Acc_all'], epoch+1)
                self.writer.add_scalar('F1/'+phase, results['F1'], epoch+1)
                self.writer.add_scalar('F2/'+phase, results['F2'], epoch+1)
                self.writer.add_scalar('F3/'+phase, results['F3'], epoch+1)
                self.writer.add_scalar('F4/'+phase, results['F4'], epoch+1)
                

                if self.mode == "eann":
                    epoch_loss_fnd = running_loss_fnd / len(self.dataloaders[phase].dataset)
                    print('Loss_fnd: {:.4f} '.format(epoch_loss_fnd))
                    epoch_loss_event = running_loss_event / len(self.dataloaders[phase].dataset)
                    print('Loss_event: {:.4f} '.format(epoch_loss_event))
                    self.writer.add_scalar('Loss_fnd/'+phase, epoch_loss_fnd, epoch+1)
                    self.writer.add_scalar('Loss_event/'+phase, epoch_loss_event, epoch+1)
                #二分类
                # if phase == 'test':
                #     if results['acc'] > best_acc_test:
                #         best_acc_test = results['acc']
                #         best_model_wts_test = copy.deepcopy(self.model.state_dict())
                #         best_epoch_test = epoch+1
                #         if best_acc_test > self.save_threshold:
                #             torch.save(self.model.state_dict(), self.save_param_path + "_test_epoch" + str(best_epoch_test) + "_{0:.4f}".format(best_acc_test))
                #             print ("saved " + self.save_param_path + "_test_epoch" + str(best_epoch_test) + "_{0:.4f}".format(best_acc_test) )
                #     else:
                #         if epoch-best_epoch_test >= self.epoch_stop-1:
                #             is_earlystop = True
                #             print ("early stopping...")
                #四分类
                if phase == 'test':
                    if results['Acc_all'] > best_acc_test:
                        best_acc_test = results['Acc_all']
                        best_model_wts_test = copy.deepcopy(self.model.state_dict())
                        best_epoch_test = epoch+1
                        if best_acc_test > self.save_threshold:
                            torch.save(self.model.state_dict(), self.save_param_path + "_test_epoch" + str(best_epoch_test) + "_{0:.4f}".format(best_acc_test))
                            print ("saved " + self.save_param_path + "_test_epoch" + str(best_epoch_test) + "_{0:.4f}".format(best_acc_test) )
                    else:
                        if epoch-best_epoch_test >= self.epoch_stop-1:
                            is_earlystop = True
                            print ("early stopping...")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on test: epoch" + str(best_epoch_test) + "_" + str(best_acc_test))

        if self.mode == "eann":
            print("Event: Best model on test: epoch" + str(best_epoch_test_event) + "_" + str(best_acc_test_event))

        self.model.load_state_dict(best_model_wts_test)
        return self.test()


    def test(self):
        since = time.time()

        self.model.cuda()
        self.model.eval()   

        pred = []
        label = []
        multilabel = []

        if self.mode == "eann":
            pred_event = []
            label_event = []

        for batch in tqdm(self.dataloaders['test']):
            with torch.no_grad(): 
                batch_data=batch
                for k,v in batch_data.items():
                    batch_data[k]=v.cuda()
                batch_label = batch_data['label']
                batch_multilabel=batch_data['multilabel']

                if self.mode == "eann":
                    batch_label_event = batch_data['label_event']
                    batch_outputs, batch_outputs_event, fea = self.model(**batch_data)
                    _, batch_preds_event = torch.max(batch_outputs_event, 1)

                    label_event.extend(batch_label_event.detach().cpu().numpy().tolist())
                    pred_event.extend(batch_preds_event.detach().cpu().numpy().tolist())
                else: 
                    batch_outputs,fea,_,_,_ = self.model(**batch_data)

                _, batch_preds = torch.max(batch_outputs, 1)

                # label.extend(batch_label.detach().cpu().numpy().tolist())
                multilabel.extend(batch_multilabel.detach().cpu().numpy().tolist())
                pred.extend(batch_preds.detach().cpu().numpy().tolist())

        
        #二分类
        # print (get_confusionmatrix_fnd(np.array(pred), np.array(label)))
        # print (metrics(label, pred))

        # if self.mode == "eann" and self.model_name != "FANVM":
        #     print ("event:")
        #     print (accuracy_score(np.array(label_event), np.array(pred_event)))

        # return metrics(label, pred)
        
        #四分类
        print (get_confusionmatrix_fnd(np.array(pred), np.array(multilabel)))
        print (evaluation4class(multilabel, pred))

        if self.mode == "eann" and self.model_name != "FANVM":
            print ("event:")
            print (accuracy_score(np.array(label_event), np.array(pred_event)))

        return evaluation4class(multilabel, pred)

