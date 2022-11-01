import pickle
import numpy as np
from util import *
from os import listdir
from os.path import isfile, join
import bisect
from collections import deque
from torch import nn
import torch
from torch import optim
from torch.autograd import Variable
import pickle
import json
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os.path
import random

DATASET_PATH = '/home/james/Desktop/research/datasets/raw-data-95-100/'

class BasicBlockDilated(nn.Module):
    def __init__(self, inplanes, planes, stage=0, block=0, kernel_size=3, numerical_names=False, 
                 stride=None, dilations=(1,1)):
        super(BasicBlockDilated, self).__init__()
        self.block = block
        if stride is None:
            if block!=0 or stage==0:
                stride=1
            else:
                stride=2

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=dilations[0],
                              bias=False, dilation=dilations[0])
        self.bn1 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, padding=dilations[1],
                              bias=False, dilation=dilations[1])

        self.bn2 = nn.BatchNorm1d(planes)

        if block == 0:
            self.conv3 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False)

            self.bn3 = nn.BatchNorm1d(planes)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.block == 0:
            identity = self.conv3(x)
            identity = self.bn3(identity)
        out += identity
        out = self.relu(out)
        
        return out

class ResNet18(nn.Module):
    def __init__(self, blocks=None, block=None, numerical_names = None):
        super(ResNet18, self).__init__()
        
        inplanes = 1
        in_channels = 64
        
        if blocks is None:
            blocks = [2, 2, 2, 2]
        if block is None:
            block = BasicBlockDilated
        if numerical_names is None:
            numerical_names = [True] * len(blocks)
        
        self.conv1 = nn.Conv1d(inplanes, 64, kernel_size=7, stride=2, bias=False, padding=3)
        
        self.bn1 = nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layers = []
        
        out_channels = 64
        
        self.dilated_layers = nn.Sequential(
        BasicBlockDilated(64, 64, 0, 0, dilations=(1,2), numerical_names = False),
        BasicBlockDilated(64, 64, 0, 1, dilations=(4,8), numerical_names = False),
        
        BasicBlockDilated(64, 128, 1, 0, dilations=(1,2), numerical_names = False),
        BasicBlockDilated(128, 128, 1, 1, dilations=(4,8), numerical_names = False),
        
        BasicBlockDilated(128, 256, 2, 0, dilations=(1,2), numerical_names = False),
        BasicBlockDilated(256, 256, 2, 1, dilations=(4,8), numerical_names = False),
        
        BasicBlockDilated(256, 512, 3, 0, dilations=(1,2), numerical_names = False),
        BasicBlockDilated(512, 512, 3, 1, dilations=(4,8), numerical_names = False)
        )
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dilated_layers(out)
    
        out = self.avgpool(out)
        
        return out

class CombinedModel(nn.Module):
    '''PyTorch implementation of Var-CNN WF attack'''
    def __init__(self, nb_classes):
        super(CombinedModel, self).__init__()
        self.resnet_time = ResNet18()
        self.resnet_dir = ResNet18()
        
        self.fc_met = nn.Linear(7, 32)
        self.bn_met = nn.BatchNorm1d(32)
        self.relu_met = nn.ReLU()
        
        self.fc = nn.Linear(1024, 1024)
        self.bn = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
        self.last_fc = nn.Linear(1024, nb_classes)
        
    def forward(self, dir_input, time_input):
        
        t = self.resnet_time(time_input)
        d = self.resnet_dir(dir_input)
        
        t = t.view((-1, 512))
        d = d.view((-1, 512))
        
        out = torch.cat((d, t), dim = 1)
        
        out = self.fc(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.last_fc(out)
        out = torch.softmax(out, dim = 1)
        
        return out

    def get_feature_vector(self, dir_input, time_input, num=50):
        t = self.resnet_time(time_input)
        d = self.resnet_dir(dir_input)

        t = t.view((-1, 512))
        d = d.view((-1, 512))

        out = torch.cat((d, t), dim = 1)

        out = self.fc(out)
        out = self.bn(out)

        return out[:num]

class AWF(nn.Module):
    '''PyTorch implementation of AWF attack (Rimmer)'''
    def __init__(self, nb_classes):
        super(AWF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(1, 32, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(19936, 500),
            nn.ReLU(),
            nn.Linear(500, nb_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.out(x)
        output = torch.softmax(output, dim=1)
        return output

class TestData(Dataset):
    def __init__(self, test_dir, test_time, test_metadata, test_labels, device):
        self.x_dir = test_dir
        self.x_time = test_time
        self.x_metadata = test_metadata
        self.y_test = test_labels
        self.device = device
            
    def __getitem__(self, index):
        x_dir_cuda = torch.from_numpy(self.x_dir[index]).reshape(1,10000).float().to(self.device)
        x_time_cuda = torch.from_numpy(self.x_time[index]).reshape(1,10000).float().to(self.device)
        x_metadata_cuda = torch.from_numpy(self.x_metadata[index]).float().to(self.device)
        y_test_cuda = torch.from_numpy(self.y_test[index]).float().to(self.device)
        
        return x_dir_cuda, x_time_cuda, x_metadata_cuda, y_test_cuda
    
    def __len__(self):
        return len(self.x_dir)
    
class ValidationData(Dataset):
    def __init__(self, val_dir, val_time, val_metadata, val_labels, device):
        self.x_dir = val_dir
        self.x_time = val_time
        self.x_metadata = val_metadata
        self.y_val = val_labels
        self.device = device
            
    def __getitem__(self, index):

        x_dir_cuda = torch.from_numpy(self.x_dir[index]).reshape(1,10000).float().to(self.device)
        x_time_cuda = torch.from_numpy(self.x_time[index]).reshape(1,10000).float().to(self.device)
        x_metadata_cuda = torch.from_numpy(self.x_metadata[index]).float().to(self.device)
        y_val_cuda = torch.from_numpy(self.y_val[index]).float().to(self.device)

        return x_dir_cuda, x_time_cuda, x_metadata_cuda, y_val_cuda
    
    def __len__(self):
        return len(self.x_dir)

class TrainData(Dataset):
    def __init__(self, train_dir, train_time, train_metadata, train_labels, device):
        self.x_dir = train_dir
        self.x_time = train_time
        self.x_metadata = train_metadata
        self.y_train = train_labels
        self.device = device
            
    def __getitem__(self, index):


        x_dir_cuda = torch.from_numpy(self.x_dir[index]).reshape(1,10000).float().to(self.device)
        x_time_cuda = torch.from_numpy(self.x_time[index]).reshape(1,10000).float().to(self.device)
        x_metadata_cuda = torch.from_numpy(self.x_metadata[index]).float().to(self.device)
        y_train_cuda = torch.from_numpy(self.y_train[index]).float().to(self.device)
        
        return x_dir_cuda, x_time_cuda, x_metadata_cuda, y_train_cuda
    
    def __len__(self):
        return len(self.x_dir)

class TorTraffic:
    '''Tor network traffic environment'''
    def __init__(self, step_size=.02, cutoff_time=50, cutoff_length=10000):
        self.current_trace = 0
        self.current_time = 0.0
        self.step_size = step_size
        self.cutoff_time = cutoff_time
        self.cutoff_length = cutoff_length
        self.defended_traces = deque(maxlen=1000)
        self.current_trace_data = ()
        self.accuracy_list = []
        self.trace_data = []
        self.action_size = 16

        #load data for Var-CNN model
        data_dir = 'sirinam_1000/'
        train_dir = np.load(data_dir + 'train_dir.npy')
        train_time = np.load(data_dir + 'train_time.npy')
        train_metadata = np.load(data_dir + 'train_metadata.npy')
        train_labels = np.load(data_dir + 'train_labels.npy')

        val_dir = np.load(data_dir + 'val_dir.npy')
        val_time = np.load(data_dir + 'val_time.npy')
        val_metadata = np.load(data_dir + 'val_metadata.npy')
        val_labels = np.load(data_dir + 'val_labels.npy')

        test_dir = np.load(data_dir + 'test_dir.npy')
        test_time = np.load(data_dir + 'test_time.npy')
        test_metadata = np.load(data_dir + 'test_metadata.npy')
        test_labels = np.load(data_dir + 'test_labels.npy')

        device = torch.device("cuda")
        self.device = device

        nb_classes = 95
        self.model = CombinedModel(nb_classes).to(device)

        train_dataset = TrainData(train_dir, train_time, train_metadata, train_labels, device)
        self.train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
        
        val_dataset = ValidationData(val_dir, val_time, val_metadata, val_labels, device)
        self.val_loader = DataLoader(val_dataset, batch_size=50, shuffle=True)

        test_dataset = TestData(test_dir, test_time, test_metadata, test_labels, device)
        self.test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

        #load data for use by RL agent
        for i in range(5000):
            direction = train_dir[i].flatten()
            time = train_time[i].flatten()
            label = train_labels[i]

            #abs_time = np.zeros(self.cutoff_length, dtype=np.float32)
            abs_time = []
            current_time = 0.0
            for i in range(10000):
                if(direction[i] == 0):
                    break
                current_time += time[i]
                abs_time.append(current_time)

            abs_time = np.array(abs_time)

            self.trace_data.append((direction, time, abs_time, label))


    def TrainEpoch(self, optimizer):
        '''Trains attack model for one epoch'''
        train_correct = 0
        total = 0
        for batch_idx, (data_dir, data_time, data_met, target) in enumerate(self.train_loader):
            optimizer.zero_grad()
            output = self.model(data_dir, data_time)
            
            pred = output.argmax(dim=1, keepdim=True)
            t = target.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(t.view_as(pred)).sum().item()
            total += len(pred)

            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print("Loss: {:0.6f}".format(loss.item()))

        print("Train accuracy: {}".format(float(train_correct)/total))


    def TrainAttack(self):
        '''Trains attack (to be used for feature extraction)'''

        if os.path.exists('var_cnn_opt.pth'):
            print('Loading Model')
            self.model = torch.load('var_cnn_opt.pth')
            return
        
        for i in range(20):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=np.sqrt(0.1), cooldown=0, min_lr=1e-5)
            self.TrainEpoch(optimizer)

            correct = 0
            total = 0
            for batch_idx, (data_dir, data_time, data_met, target) in enumerate(self.val_loader):
                output = self.model(data_dir, data_time)
                pred = output.argmax(dim=1, keepdim=True)
                t = target.argmax(dim=1, keepdim=True)
                correct += pred.eq(t.view_as(pred)).sum().item()
                total += len(pred)

            val_acc = float(correct) / total
            print('Validation Accuracy: {}'.format(val_acc))

            scheduler.step(val_acc)

        torch.save(self.model, 'var_cnn_opt.pth')


    def getMeanFE(self):
        '''Gets the mean feature extraction vector'''

        with torch.no_grad():
            fv_list = []
            sizing = []
            large_traces = []
            large_fv_list = []

            for batch_idx, (data_dir, data_time, data_met, target) in enumerate(self.val_loader):
                for i in range(50):
                    #may change cutoff later, but for now take traces with over 2500 packets
                    if np.count_nonzero(data_dir.cpu().clone().numpy()[i]) > 3500:
                        large_traces.append((data_dir.cpu().clone().numpy()[i], data_time.cpu().clone().numpy()[i], data_met.cpu().clone().numpy()[i]))

            large_traces = large_traces[:300]
            data_dir_list = np.asarray([x[0] for x in large_traces])
            data_time_list = np.asarray([x[1] for x in large_traces])
            data_met_list = np.asarray([x[2] for x in large_traces])

            for i in range(6):
                data_dir = torch.from_numpy(data_dir_list[i*50:(i+1)*50]).float().reshape(50,1,10000).float().to(self.device)
                data_time = torch.from_numpy(data_time_list[i*50:(i+1)*50]).float().reshape(50,1,10000).float().to(self.device)
                #data_met = torch.from_numpy(data_met_list[i*50:(i+1)*50]).float().reshape(50,7).float().to(self.device)

                self.model.eval()
                fv = self.model.get_feature_vector(data_dir, data_time)
                fv.cpu()
                large_fv_list.append(fv)


            tensor_list = torch.cat(large_fv_list)
            mean_list = torch.mean(tensor_list, 0)

        self.mean_fv = mean_list.cpu().clone().numpy()
        return mean_list.cpu().clone().numpy()

    def get_obs_vector(self):
        '''Returns a vector representing that traffic that has been sent at certain point in time'''
        
        direction, time_gap, abs_time, label = self.current_trace_data

        #show packets sent before current time
        index = np.searchsorted(abs_time, self.current_time)

        direction = direction[:index]
        time = time_gap[:index]

        direction = np.pad(direction, pad_width=(0, 10000 - len(direction)), mode='constant')
        time = np.pad(time, pad_width=(0, 10000 - len(time)), mode='constant')

        return (direction, time, label)

    def step(self, action):

        #get original feature extraction vector
        if(self.current_time == 0.0):
            dir_seq, time_seq, _,  _ = self.current_trace_data
            self.model.eval()
            fv = self.get_FE(dir_seq, time_seq)
            self.orig_distance = np.linalg.norm(self.mean_fv - fv)
            self.last_distance = self.orig_distance

        self.current_time += self.step_size
        website = self.trace_data[self.current_trace][3]
    
        #use action to insert packets
        direction, timing, abs_timing, label = self.current_trace_data
        if(action == 0):
            #don't pad
            pass
        elif(action <= 10):
            #add 1-10 download packets at current time
            for i in range(action):
                #get index from timing
                index = np.searchsorted(abs_timing, self.current_time)
                abs_timing = np.insert(abs_timing, index, self.current_time)[:10000]
                #check to make sure this is correct
                timing = np.insert(timing, index, self.current_time-timing[index-1])[:10000]
                direction = np.insert(direction, index, -1)[:10000]

        elif(action > 10 and action <= 15):
            #add 1-5 upload packets
            for i in range(action-10):
                index = np.searchsorted(abs_timing, self.current_time)
                abs_timing = np.insert(abs_timing, index, self.current_time)[:10000]
                timing = np.insert(timing, index, self.current_time-timing[index-1])[:10000]
                direction = np.insert(direction, index, 1)[:10000]
        
        self.current_trace_data = direction, timing, abs_timing, label
        
        if(self.current_time >= self.cutoff_time):
            is_done = True
            self.current_trace += 1
            self.current_time = 0.0
        else:
            is_done = False

        if(self.current_trace >= len(self.trace_data)):
            self.current_trace = 0

        return self.get_obs_vector(), is_done, website, self.current_trace

    def reset(self):
        self.current_time = 0.0

        self.current_trace_data = self.trace_data[self.current_trace]

        return self.get_obs_vector()
    
    def get_FE(self, dir_seq, time_seq):
        '''Gets the feature extractor for an individual trace'''

        with torch.no_grad():
            self.model.eval()
            dir_cuda = torch.from_numpy(dir_seq).reshape(1,1,10000).float().to(self.device)
            dir_cuda = torch.cat([dir_cuda]*50)
            time_cuda = torch.from_numpy(time_seq).reshape(1,1,10000).float().to(self.device)
            time_cuda = torch.cat([time_cuda]*50)
            
            fv = self.model.get_feature_vector(dir_cuda, time_cuda, num=1)
        '''
        with torch.no_grad():
            self.model.eval()
            fv = self.model(dir_seq, time_seq)
        '''

        return fv.cpu().clone().numpy()

    def get_rewards(self, states):
        '''Gets states, calculates feature vectors and rewards'''
        direction_list = []
        time_list = []
        for i in range(1,51):
            direction, time, label = states[i]
            direction_list.append(torch.from_numpy(direction).reshape(1,1,10000).float().to(self.device))
            time_list.append(torch.from_numpy(time).reshape(1,1,10000).float().to(self.device))

        dir_cuda = torch.cat(direction_list)
        time_cuda = torch.cat(time_list)
        
        with torch.no_grad():
            self.model.eval()
            fv = self.model.get_feature_vector(dir_cuda, time_cuda, num=50).cpu().clone().numpy()

        rewards = []
        for i in range(50):
            distance = np.linalg.norm(self.mean_fv - fv[i])
            reward = self.last_distance - distance
            self.last_distance = distance
            rewards.append(reward)

        return rewards

    def retrain_attack(self):
        #TODO
        pass

#env = TorTraffic()
#env.TrainAttack()
#env.getMeanFE()
#env.reset()
#env.step(3)
