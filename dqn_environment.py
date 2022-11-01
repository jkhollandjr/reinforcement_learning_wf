import pickle
import numpy as np
from util import *
from os import listdir
from os.path import isfile, join
import bisect
from collections import deque

#There are a variety of ways to do this..including CNNs
#For now, I'll focus on using a lookback lengths

LOOKBACK_SIZE = [1, 2, 4, 8, 12, 16, 20, 24, 32, 50]
#ACTION_SIZE = [.0025, .0033, .005, .01, .02, .04, .0834, .167, .33, 1.0]
ACTION_SIZE = [1.0, .33, .167, .0834, .04, .02, .01, .005, .0033, .0025]
DATASET_PATH = '/home/james/Desktop/research/datasets/wang_knn/closed_world/'

#env.step(action) should return next_obs, is_done, website
#Object-oriented approach is probably best here

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from data_utils import *
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adamax

#Tik-Tok attack model
class ConvNet:
    @staticmethod
    def build(classes,
              input_shape,
              activation_function=("elu", "relu", "relu", "relu", "relu", "relu"),
              dropout=(0.1, 0.1, 0.1, 0.1, 0.5, 0.7),
              filter_num=(32, 64, 128, 256),
              kernel_size=8,
              conv_stride_size=1,
              pool_stride_size=4,
              pool_size=8,
              fc_layer_size=(512, 512)):

        # confirm that parameter vectors are acceptable lengths
        assert len(filter_num) + len(fc_layer_size) <= len(activation_function)
        assert len(filter_num) + len(fc_layer_size) <= len(dropout)

        # Sequential Keras model template
        model = Sequential()

        # add convolutional layer blocks
        for block_no in range(0, len(filter_num)):
            if block_no == 0:
                model.add(Conv1D(filters=filter_num[block_no],
                                 kernel_size=kernel_size,
                                 input_shape=input_shape,
                                 strides=conv_stride_size,
                                 padding='same',
                                 name='block{}_conv1'.format(block_no)))
            else:
                model.add(Conv1D(filters=filter_num[block_no],
                                 kernel_size=kernel_size,
                                 strides=conv_stride_size,
                                 padding='same',
                                 name='block{}_conv1'.format(block_no)))

            model.add(BatchNormalization())

            model.add(Activation(activation_function[block_no], name='block{}_act1'.format(block_no)))

            model.add(Conv1D(filters=filter_num[block_no],
                             kernel_size=kernel_size,
                             strides=conv_stride_size,
                             padding='same',
                             name='block{}_conv2'.format(block_no)))

            model.add(BatchNormalization())

            model.add(Activation(activation_function[block_no], name='block{}_act2'.format(block_no)))

            model.add(MaxPooling1D(pool_size=pool_size,
                                   strides=pool_stride_size,
                                   padding='same',
                                   name='block{}_pool'.format(block_no)))

            model.add(Dropout(dropout[block_no], name='block{}_dropout'.format(block_no)))

        # flatten output before fc layers
        model.add(Flatten(name='flatten'))

        # add fully-connected layers
        for layer_no in range(0, len(fc_layer_size)):
            model.add(Dense(fc_layer_size[layer_no],
                            kernel_initializer=glorot_uniform(seed=0),
                            name='fc{}'.format(layer_no)))

            model.add(BatchNormalization())
            model.add(Activation(activation_function[len(filter_num)+layer_no],
                                 name='fc{}_act'.format(layer_no)))

            model.add(Dropout(dropout[len(filter_num)+layer_no],
                              name='fc{}_drop'.format(layer_no)))

        # add final classification layer
        model.add(Dense(classes, kernel_initializer=glorot_uniform(seed=0), name='fc_final'))
        model.add(Activation('softmax', name="softmax"))

        # compile model with Adamax optimizer
        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        return model

class TorTraffic:
    def __init__(self, step_size=1, cutoff_time=50, cutoff_length=5000, lookback=LOOKBACK_SIZE):
        self.current_trace = 0
        self.current_time = 0.0
        self.lookback = lookback
        self.step_size = step_size
        self.cutoff_time = cutoff_time
        self.cutoff_length = cutoff_length
        self.defended_traces = deque(maxlen=1000)
        self.action_size = len(ACTION_SIZE)
        self.current_trace_data = ()
        self.accuracy_list = []

        self.trace_data = []

        file_list = [f for f in listdir(DATASET_PATH) if isfile(join(DATASET_PATH, f))]
        for file_name in file_list:
            trace = get_trace(DATASET_PATH + str(file_name), self.cutoff_time, self.cutoff_length)
            website = int(file_name.split('-')[0])
            trace_num = int(file_name.split('-')[1])

            download_packets = get_download_packets(trace)
            upload_packets = get_upload_packets(trace)

            self.trace_data.append((website, trace_num, download_packets, upload_packets)) 

        print("Dataset Loaded!")

        #train initial attack model
        X, y = load_data(DATASET_PATH, 1)
        count = len(list(y))
        train_end = int(count * .8)
        val_end = int(count * .9)

        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]

        #convert class vectors to binary
        classes = len(set(list(y_train)))
        y_train = to_categorical(y_train, classes)
        y_val = to_categorical(y_val, classes)
        y_test = to_categorical(y_test, classes)

        self.model = ConvNet.build(classes=classes, input_shape=(5000, 1))

        checkpoint = ModelCheckpoint('tt_model.ckpt', monitor='val_loss', save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='auto', restore_best_weights=True)
        callbacks_list = [checkpoint, early_stopping]

        model = load_model('tt_model_knn_data')
        #history = self.model.fit(X_train, y_train, epochs=40, verbose=2, validation_data=(X_val, y_val), callbacks=callbacks_list)
        #self.model.save('tt_model_knn_data')


    def get_obs_vector(self):
        obs_v = []

        _, _, download_packets, upload_packets = self.current_trace_data
        #get download packet counts
        for l in self.lookback:
            counter = 0
            for time in download_packets:
                if(time > (self.current_time - l) and time < self.current_time):
                    counter += 1
                elif(time > self.current_time):
                    break
            obs_v.append(float(counter))

        #get upload packet counts
        for l in self.lookback:
            counter = 0
            for time in upload_packets:
                if(time > (self.current_time - l) and time < self.current_time):
                    counter += 1
                else:
                    break
            obs_v.append(float(counter))
        
        return obs_v


    def step(self, action):
        #action is the packet sending rate

        self.current_time += self.step_size

        segment_start = self.current_time - self.step_size
        segment_end = self.current_time


        download_trace_segment = list(filter(lambda x: x > segment_start and x < segment_end, self.trace_data[self.current_trace][2]))
        upload_trace_segment = list(filter(lambda x: x > segment_start and x < segment_end, self.trace_data[self.current_trace][3]))
        original_bandwidth = len(download_trace_segment) + len(upload_trace_segment)

        website = self.trace_data[self.current_trace][0]
        trace_num = self.trace_data[self.current_trace][1]

        defended_download_segment = self.defend_segment(download_trace_segment, action, segment_start)
        defended_upload_segment = self.defend_segment(upload_trace_segment, action, segment_start)
        defended_bandwidth = len(defended_download_segment) + len(defended_upload_segment)

        added_packets = defended_bandwidth - original_bandwidth

        #calculate reward from bandwidth
        bw_reward = float(added_packets) / (-1*2000.0)

        website, trace_num, current_download_trace, current_upload_trace = self.current_trace_data
        self.current_trace_data = (website, trace_num, current_download_trace + defended_download_segment, current_upload_trace + defended_upload_segment)

        acc_reward = 0.0

        obs_v = self.get_obs_vector()
        if(self.current_time >= self.cutoff_time):
            is_done = True
            acc_reward = self.calculate_attack_reward()
            self.current_trace += 1
            self.current_time = 0.0
        else:
            is_done = False

        if(self.current_trace >= len(self.trace_data)):
            self.current_trace = 0
        
        reward = acc_reward + bw_reward

        return obs_v, is_done, website, trace_num, self.current_trace, reward
        
    def reset(self):
        #self.current_trace = 0
        self.current_time = 0.0 
        self.current_trace_data = self.trace_data[self.current_trace]

        return self.get_obs_vector()

    def defend_segment(self, trace_segment, action, start_time): 
        time = start_time
        if(len(trace_segment) == 0):
            trace_segment.append(start_time)
        #trace_segment.append(time)
        #trace_segment.sort()
        while(time < (start_time + self.step_size)):
            last_packet_index = bisect.bisect(trace_segment, time)
            last_packet = trace_segment[last_packet_index-1]

            packet_gap = time - last_packet
            action_gap = ACTION_SIZE[action]
            #print("{}\t{}\n".format(download_gap, action_gap))
            if(packet_gap >= action_gap):
                trace_segment.append(time)
                #this probably isn't particularly efficient, replace with bisect function later
                trace_segment.sort()

            time += .001

        return trace_segment


    def calculate_attack_reward(self):
        website, trace_num, download_packets, upload_packets = self.current_trace_data

        download_packets = [-p for p in download_packets]
        defended_trace = sorted(download_packets + upload_packets, key=lambda x: abs(x))

        if(len(defended_trace) > self.cutoff_length):
            defended_trace = defended_trace[:self.cutoff_length]
        else:
            defended_trace += [0]*(self.cutoff_length - len(defended_trace))

        self.defended_traces.append((website, defended_trace))
        #if(np.random.randint(0,50)==10):
            #print(defended_trace)

        defended_trace_array = np.expand_dims(np.asarray(defended_trace), axis=1)
        pred_website = self.model.predict(np.expand_dims(defended_trace_array, axis=0))
        #print("{}\t{}\n".format(np.argmax(pred_website[0]), website))
        if(website == np.argmax(pred_website[0])):
            acc_reward = 0
            self.accuracy_list.append(1)
        else:
            acc_reward = 3
            self.accuracy_list.append(0)

        return acc_reward

    def calculate_reward(self, original_trace, actions):
        trace = self.trace_data[original_trace]
        website = trace[0]
        download_packets = trace[2]
        upload_packets = trace[3]
        original_bandwidth = len(download_packets) + len(upload_packets)

        #create defended trace
        #problem: need two different actions - for upload and download!
        #for now, I'll just assume that upload mimic download (which is mostly true)

        time = 0.0
        download_packets.append(time)
        download_packets.sort()
        while(time < download_packets[-1] and time < self.cutoff_time):
            last_download_index = bisect.bisect(download_packets, time)
            last_packet = download_packets[last_download_index-1]

            download_gap = time - last_packet
            action = actions[int(time/self.step_size)]
            action_gap = ACTION_SIZE[action]
            #print("{}\t{}\n".format(download_gap, action_gap))
            if(download_gap >= action_gap):
                download_packets.append(time)
                #this probably isn't particularly efficient, replace with bisect function later
                download_packets.sort()

            time += .001

        time = 0.0
        download_packets.append(time)
        download_packets.sort()
        while(time < upload_packets[-1] and time < self.cutoff_time):
            last_upload_index = bisect.bisect(upload_packets, time)
            last_packet = upload_packets[last_upload_index-1]

            upload_gap = time - last_packet
            action = actions[int(time/self.step_size)]
            action_gap = ACTION_SIZE[action]
            #magic number warning
            action_gap *= 4
            if(upload_gap >= action_gap):
                upload_packets.append(time)
                upload_packets.sort()

            time += .001

        padded_bandwidth = len(download_packets) + len(upload_packets)

        bandwidth_ratio = float(padded_bandwidth) / float(original_bandwidth)
        #print("bw ratio: {}".format(bandwidth_ratio))
        bandwidth_reward = 1.0 / bandwidth_ratio 

        #run attack

        #convert upload/download traces to numpy array with +/- timestamps
        download_packets = [-p for p in download_packets]
        defended_trace = sorted(download_packets + upload_packets, key=lambda x: abs(x))

        if(len(defended_trace) > self.cutoff_length):
            defended_trace = defended_trace[:self.cutoff_length]
        else:
            defended_trace += [0]*(self.cutoff_length - len(defended_trace))

        self.defended_traces.append((website, defended_trace))


        defended_trace_array = np.expand_dims(np.asarray(defended_trace), axis=1)
        pred_website = self.model.predict(np.expand_dims(defended_trace_array, axis=0))
        #print("{}\t{}\n".format(np.argmax(pred_website[0]), website))
        if(website == np.argmax(pred_website[0])):
            acc_reward = 0
        else:
            acc_reward = 2

        reward = acc_reward + bandwidth_reward

        return reward

    def retrain_attack(self):
        
        X_train = np.asarray([np.expand_dims(np.asarray(x[1]),axis=1) for x in self.defended_traces])
        y_train = np.asarray([x[0] for x in self.defended_traces])
        y_train = to_categorical(y_train, 100)

        self.model.fit(X_train, y_train, epochs=10, verbose=2)


