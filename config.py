# -*- coding: utf-8 -*-
import Model_new_user_attention3 as Model

dataset = ['last_music', 'gowalla']
min_length = [100, 100]  # more than
max_length = [1000, 500,]  # less than
user_count_list = [992, 2000]
top_n_item_list = [30000, 25000, 20000, 15000, 10000]

hist_length_list = [1, 2, 3, 4, 5, 6, 7]

# =======================================
# lastfm
MHPE_lastfm_model = Model
MHPE_lastfm_emb_size = 256
MHPE_lastfm_neg_size = 5
MHPE_lastfm_hist_len = 3
MHPE_lastfm_cuda = "0"
MHPE_lastfm_learning_rate = 0.001
MHPE_lastfm_decay = 0.01
MHPE_lastfm_batch_size = 1024
MHPE_lastfm_test_and_save_step = 20
MHPE_lastfm_epoch_num = 200


# =======================================
# gowalla
MHPE_gowalla_model = Model
MHPE_gowalla_emb_size = 256
MHPE_gowalla_neg_size = 5
MHPE_gowalla_hist_len = 3
MHPE_gowalla_cuda = "2"
MHPE_gowalla_learning_rate = 0.001
MHPE_gowalla_decay = 0.001
MHPE_gowalla_batch_size = 512
MHPE_gowalla_test_and_save_step = 20
MHPE_gowalla_epoch_num = 200
