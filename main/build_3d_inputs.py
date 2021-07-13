import wfdb
import numpy as np
import scipy.io as sio
import os
import json
import pickle
import random
import wfdb
np.set_printoptions(suppress = True)
from scipy import signal
from scipy.signal import butter, lfilter, freqz
from scipy.signal import medfilt
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import shutil

def data_prepare_single_128samples_beat_4classes(data_dir, data_list, save_dir):
    '''参考论文：Automated Heartbeat Classification Using 3-D Inputs Based on Convolutional Neural
                Network With Multi-Fields of View
       心跳分段：将mitdb导联II的数据单个beat(从前1个R峰后的0.14s处开始,到当前R峰之后的0.28s结束,采样率位360Hz)
       得到的每个beat长度不等，因而resample到长度为M=128的beat


       每个样本由current heartbeat、pre_RR_tatio、near_pre_RR_ratio组成，shape(3,128,1)
       '''
    folder_name = save_dir.split('/')[2] + '/'
    Heartbeats_img = '../exp1/HeartBeatsImg/' + folder_name
    if os.path.exists(Heartbeats_img):
        shutil.rmtree(Heartbeats_img)
    if not os.path.exists(Heartbeats_img):
        os.makedirs(Heartbeats_img)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fs = 360#257、128
    left_beat = int(fs * 0.14)#0.14
    right_beat = int(fs * 0.28)#0.28
    hrv_length = 5 - 1
    feature_num = 3
    channel = 1
    # left_beat = 70
    # right_beat = 99
    beat_length = 128
    '''not AAMI standard, but most papers like this'''
    label_t4 = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
                'A': 'S', 'a':'S', 'J':'S', 'S': 'S',
                'V': 'V', 'E': 'V',
                'F': 'F',
                }


    num = 0
    for cur_record_id in data_list:
        print(cur_record_id)
        record_info = wfdb.rdsamp(os.path.join(data_dir, str(cur_record_id)))
        record_ann = wfdb.rdann(os.path.join(data_dir, str(cur_record_id)), 'atr')

        samples = record_ann.annsamp
        symbols = record_ann.anntype

        MLII_index = record_info.signame.index('MLII')#'MLII'、'II'、'ECG1'
        # sig = record_info.p_signals[:,0:1]
        sig = record_info.p_signals[:, MLII_index]
        sig = sig.reshape(1, len(sig))

        # sig = wtmedian_denoise(sig[0], gain_mask = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0 , 0])
        # sig = sig.reshape(1, len(sig))

        # sig = (sig - np.mean(sig)) / np.std(sig)
        sig_length = sig.shape[1]
        print('sig shape:', sig.shape)


        ECG_Data = np.array([]).reshape((0, feature_num, beat_length, channel))
        Label = np.array([]).reshape((0, 4))

        beat_label = []
        r_position = []
        for i, symbol in enumerate(symbols):
            #对不属于这16类的心跳，在计算平均RR间期时也需要考虑，不能跳过这些心跳
            r_position.append(samples[i])
            if symbol in label_t4.keys():
                beat_label.append(symbol)
        rr_interval = np.diff(r_position)
        mean_rr = np.mean(rr_interval)
        # rr_interval = (rr_interval - np.min(rr_interval))/(np.max(rr_interval) - np.min(rr_interval ))
        # rr_interval = rr_interval / mean_rr
        print('len beat_label:', len(beat_label))
        # print('mean_rr:',mean_rr )
        # left_beat = int(mean_rr * 0.6)  # 0.14
        # right_beat = int(mean_rr * 0.3)  # 0.28

        # for i, label in enumerate(beat_label):
        for i, symbol in enumerate(symbols):
            '''这样做还是存在相邻的心跳不属于这15类、但划分的心跳包含部分这些不在15类中的心跳信息的可能'''
            if i >= 2 and i <= len(symbols)-1 and symbol in label_t4.keys():
            # if i >= 1  and symbol in label_t5.keys():
                #截取两个心跳的操作
                if i == 2:
                    ##注意：第1个标注samples[0]并非R峰
                    # one_beat = sig[0, :samples[i]+ right_beat + 1]
                    one_beat = sig[0, samples[i - 1] + left_beat:samples[i] + right_beat + 1]
                    one_beat = one_beat - np.mean(one_beat)
                    one_beat = signal.resample(one_beat, beat_length)
                    one_beat = one_beat.reshape(1, one_beat.shape[0])
                    pre_rr_ratio = rr_interval[i - 1] / np.mean(rr_interval[:i])
                    pre_rr_ratio = np.tile(pre_rr_ratio, (1, beat_length))
                    near_pre_rr_ratio = pre_rr_ratio


                if samples[i - 1] + left_beat < samples[i] and samples[i] + right_beat + 1 < sig_length:
                     one_beat = sig[0, samples[i - 1] + left_beat:samples[i] + right_beat + 1]
                     one_beat = one_beat - np.mean(one_beat)
                     one_beat = signal.resample(one_beat, beat_length)
                     one_beat = one_beat.reshape(1, one_beat.shape[0])
                     pre_rr_ratio = rr_interval[i - 1] / np.mean(rr_interval[:i])
                     pre_rr_ratio = np.tile(pre_rr_ratio, (1, beat_length))
                     if i <= 12:
                         near_pre_rr_ratio = pre_rr_ratio
                     else:
                         near_pre_rr_ratio = rr_interval[i - 1] / np.mean(rr_interval[i - 10:i])
                         near_pre_rr_ratio = np.tile(near_pre_rr_ratio, (1, beat_length))


                if num == 0 and i <= 11:
                    '''保存第一个record的前10个Beats'''
                    plt.figure()
                    plt.plot(one_beat[0])
                    plt.grid(True)
                    plt.savefig(Heartbeats_img + str(cur_record_id) + '_' + str(i) + '.png')

                ecg_data = np.concatenate((one_beat, pre_rr_ratio, near_pre_rr_ratio),axis=0)
                ecg_data = ecg_data.reshape(1, feature_num, beat_length, channel)
                ECG_Data = np.concatenate((ECG_Data, ecg_data), axis=0)
                label = symbol
                if label_t4[label] == 'N':
                    cur_label = np.array([[1, 0, 0, 0, ]])
                elif label_t4[label]  == 'S':
                    cur_label = np.array([[0, 1, 0, 0, ]])
                elif label_t4[label]  == 'V':
                    cur_label = np.array([[0, 0, 1, 0, ]])
                elif label_t4[label]  == 'F':
                    cur_label = np.array([[0, 0, 0, 1,]])

                Label = np.concatenate((Label, cur_label), axis=0)

        print(ECG_Data.shape)
        print(Label.shape)


        # np.savez(save_dir+str(cur_record_id), ECG_Data=ECG_Data, RR_feature=RR_feature, Label=Label)
        np.savez(save_dir + str(cur_record_id), ECG_Data=ECG_Data, Label=Label)
        num = num + 1
        # # if num == 1:
        # #     break

        print('=====Processing第'+str(num)+'个record=====\n')

if __name__ == "__main__":
    data_dir = '../../MIT-BIH_arrythimia/'
    train_record_list = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
                         205, 207, 208, 209, 215, 220, 223, 230]
    test_record_list = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
                        219, 221, 222, 228, 231, 232, 233, 234]
    train_save_dir = '../EMBC/mitdb_DS1_3d_inputs_4classes/'
    test_save_dir = '../EMBC/mitdb_DS2_3d_inputs_4classes/'
    data_prepare_single_128samples_beat_4classes(data_dir, train_record_list, train_save_dir)
    data_prepare_single_128samples_beat_4classes(data_dir, test_record_list, test_save_dir)