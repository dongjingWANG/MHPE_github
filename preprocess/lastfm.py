# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import pickle
import os

import numpy as np
import copy


def generate_data(user_count, top_n_item, min_length, max_length, data, BASE_DIR, DATA_SOURCE, small_data=False):
    tr_user_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                            'tr-user-item-time-top{}-min{}-max{}-{}'.format(top_n_item, min_length,
                                                                                            max_length,
                                                                                            NORM_METHOD) + '.lst')
    te_user_old_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                'te-user-old-item-time-top{}-min{}-max{}-{}'.format(top_n_item,
                                                                                                    min_length,
                                                                                                    max_length,
                                                                                                    NORM_METHOD) + '.lst')
    te_user_new_item_time_record = os.path.join(BASE_DIR, DATA_SOURCE,
                                                'te-user-new-item-time-top{}-min{}-max{}-{}'.format(top_n_item,
                                                                                                    min_length,
                                                                                                    max_length,
                                                                                                    NORM_METHOD) + '.lst')


    index2item_path = os.path.join(BASE_DIR, DATA_SOURCE, 'last_music_index2item_top' + str(top_n_item))
    item2index_path = os.path.join(BASE_DIR, DATA_SOURCE, 'last_music_item2index_top' + str(top_n_item))
    out_tr_uit = open(tr_user_item_time_record, 'w', encoding='utf-8')
    out_te_old_uit = open(te_user_old_item_time_record, 'w', encoding='utf-8')
    out_te_new_uit = open(te_user_new_item_time_record, 'w', encoding='utf-8')


    if os.path.exists(index2item_path) and os.path.exists(item2index_path):
        index2item = pickle.load(open(index2item_path, 'rb'))
        item2index = pickle.load(open(item2index_path, 'rb'))
        print('Total music and user %d' % len(index2item))
    else:
        print('Build index2item')
        sorted_user_series = data.groupby(['userid']).size().sort_values(ascending=False)
        print('sorted_user_series size is: {}'.format(len(sorted_user_series)))
        user_index2item = sorted_user_series.keys().tolist()

        sorted_item_series = data.groupby(['tran_name_id']).size().sort_values(ascending=False)
        print('sorted_item_series size is: {}'.format(len(sorted_item_series)))
        item_index2item = sorted_item_series.head(top_n_item).keys().tolist()
        print('item_index2item size is: {}'.format(len(item_index2item)))

        new_user_index2item = [('user_' + str(x)) for x in user_index2item]
        index2item = item_index2item + new_user_index2item
        print('index2item size is: {}'.format(len(index2item)))

        print('Most common song is "%s":%d' % (index2item[0], sorted_item_series[0]))
        print('Most active user is "%s":%d' % (index2item[top_n_item], sorted_user_series[0]))

        print('build item2index')
        item2index = dict((v, i) for i, v in enumerate(index2item))
        pickle.dump(index2item, open(index2item_path, 'wb'))
        pickle.dump(item2index, open(item2index_path, 'wb'))

    print('start loop')
    count = 0
    valid_user_count = 0
    user_group = data.groupby(['userid'])
    total = len(user_group)
    for user_id, length in user_group.size().sort_values().iteritems():
        if count % 100 == 0:
            print("=====count %d/%d======" % (count, total))
            print('%s %d' % (user_id, length))
        count += 1
        temp_user_data = user_group.get_group(user_id)
        old_time_seq = copy.deepcopy(pd.to_datetime(temp_user_data['timestamp']))
        temp_user_data.loc[:, 'timestamp_new'] = old_time_seq
        user_data = temp_user_data.sort_values(by='timestamp_new')
        music_seq = user_data['tran_name_id']
        time_seq = user_data['timestamp_new']
        time_seq = time_seq[music_seq.notnull()]
        music_seq = music_seq[music_seq.notnull()]
        delta_time = pd.to_datetime(time_seq).diff(-1).astype('timedelta64[s]') * -1
        item_seq = music_seq.apply(lambda x: (item2index[x]) if pd.notnull(x) and x in item2index else -1).tolist()

        delta_time = delta_time.tolist()
        delta_time[-1] = 0


        if NORM_METHOD == 'log':
            delta_time = np.log(np.array(delta_time) + 1.0 + 1e-6)
        elif NORM_METHOD == 'mm':
            temp_delta_time = np.array(delta_time)
            min_delta = temp_delta_time.min()
            max_delta = temp_delta_time.max()
            delta_time = (np.array(delta_time) - min_delta) / (max_delta - min_delta)
        elif NORM_METHOD == 'hour':
            delta_time = np.array(delta_time) / 3600

        time_accumulate = [0]
        for delta in delta_time[:-1]:
            next_time = time_accumulate[-1] + delta
            time_accumulate.append(next_time)

        new_item_seq = []
        new_time_accumulate = []
        valid_count = 0
        for i in range(len(item_seq)):
            if item_seq[i] != -1:
                new_item_seq.append(item_seq[i])
                new_time_accumulate.append(time_accumulate[i])
                valid_count += 1
            if valid_count >= max_length:
                break

        if len(new_item_seq) < min_length:
            continue
        else:
            valid_user_count += 1
            user_index = item2index['user_' + user_id]
            index_hash_remaining = user_index % 10
            if index_hash_remaining < 2:
                half_index = int(len(new_item_seq) / 2)
                for i in range(half_index):
                    out_tr_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\n')
                for i in range(half_index, int(len(new_item_seq))):
                    out_te_old_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\n')
                    out_te_new_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\n')


            else:
                for i in range(len(new_item_seq)):
                    out_tr_uit.write(
                        str(user_index) + '\t' + str(new_item_seq[i]) + '\t' + str(new_time_accumulate[i]) + '\n')


    print("valid_user_count is: {}".format(valid_user_count))
    out_tr_uit.close()
    out_te_old_uit.close()
    out_te_new_uit.close()


if __name__ == '__main__':
    BASE_DIR = ''
    DATA_SOURCE = 'last_music'
    NORM_METHOD = 'hour'
    top_n_item_list = [10000, 15000, 20000, 25000, 30000]

    user_count = 992

    min_length = 100
    max_length = 1000

    path = os.path.join(BASE_DIR, DATA_SOURCE, 'userid-timestamp-artid-artname-traid-traname.tsv')

    print("start reading csv")
    data = pd.read_csv(path, sep='\t',
                       error_bad_lines=False,
                       header=None,
                       names=['userid', 'timestamp', 'artid', 'artname', 'traid', 'tranname'],
                       quotechar=None, quoting=3)
    print("finish reading csv")
    data['tran_name_id'] = data['tranname'] + data['traid']
    data['art_name_id'] = data['artname'] + data['artid']
    for top_n_item in top_n_item_list:
        print("starting processing for top_n_item = {}".format(top_n_item))
        generate_data(user_count, top_n_item, min_length, max_length, data, BASE_DIR, DATA_SOURCE, small_data=False)
