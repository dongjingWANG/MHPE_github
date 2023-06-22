# -*- coding: utf-8 -*-
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import sys
from DataSet import DataSetTrain, DataSetTestNext, DataSetTestNextNew

import logging

FType = torch.FloatTensor
LType = torch.LongTensor

FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

NORM_METHOD = 'hour'


class HTSER_a:
    def __init__(self, file_path_tr, file_path_te_old, file_path_te_new, emb_size=128, neg_size=10,
                 hist_len=2,
                 user_count=992, item_count=5000, directed=True, learning_rate=0.001, decay=0.001, batch_size=1024,
                 test_and_save_step=50, epoch_num=100, top_n=30, sample_time=3, sample_size=100,
                 use_hist_attention=True, use_user_pref_attention=True, num_workers=0):
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.user_count = user_count
        self.item_count = item_count

        self.lr = learning_rate
        self.decay = decay
        self.batch = batch_size
        self.test_and_save_step = test_and_save_step
        self.epochs = epoch_num

        self.top_n = top_n
        self.sample_time = sample_time
        self.sample_size = sample_size

        self.directed = directed
        self.use_hist_attention = use_hist_attention
        self.use_user_pref_attention = use_user_pref_attention

        self.num_workers = num_workers

        self.temp_value1 = 0.0
        self.temp_value2 = 0.0

        logging.info('emb_size: {}'.format(emb_size))
        logging.info('neg_size: {}'.format(neg_size))
        logging.info('hist_len: {}'.format(hist_len))
        logging.info('user_count: {}'.format(user_count))
        logging.info('item_count: {}'.format(item_count))
        logging.info('lr: {}'.format(learning_rate))
        logging.info('epoch_num: {}'.format(epoch_num))
        logging.info('test_and_save_step: {}'.format(test_and_save_step))
        logging.info('batch: {}'.format(batch_size))
        logging.info('top_n: {}'.format(top_n))
        logging.info('sample_time: {}'.format(sample_time))
        logging.info('sample_size: {}'.format(sample_size))
        logging.info('directed: {}'.format(directed))
        logging.info('use_hist_attention: {}'.format(use_hist_attention))
        logging.info('use_user_pref_attention: {}'.format(use_user_pref_attention))

        self.data_tr = DataSetTrain(file_path_tr, user_count=self.user_count, item_count=self.item_count,
                                    neg_size=self.neg_size, hist_len=self.hist_len, directed=self.directed)
        self.data_te_old = DataSetTestNext(file_path_te_old, user_count=self.user_count, item_count=self.item_count,
                                           hist_len=self.hist_len, user_item_dict=self.data_tr.user_item_dict,
                                           directed=self.directed)
        self.data_te_new = DataSetTestNextNew(file_path_te_new, user_count=self.user_count, item_count=self.item_count,
                                              hist_len=self.hist_len, user_item_dict=self.data_tr.user_item_dict,
                                              directed=self.directed)

        self.node_dim = self.data_tr.get_node_dim()
        self.node_emb = torch.tensor(
            np.random.uniform(-0.5 / self.emb_size, 0.5 / self.emb_size, size=(self.node_dim, self.emb_size)),
            dtype=torch.float)
        self.delta = torch.ones(self.node_dim, dtype=torch.float)
        self.weight = torch.tensor(
            np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=(self.emb_size, self.emb_size)), dtype=torch.float)
        self.bias = torch.tensor(
            np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=self.emb_size), dtype=torch.float)

        self.long_short_pref_weight = torch.tensor(
            np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=(2*self.emb_size, 2)),
            dtype=torch.float)
        self.long_short_pref_bias = torch.tensor(np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=2),
                                                 dtype=torch.float)

        if torch.cuda.is_available():
            self.node_emb = self.node_emb.cuda()
            self.delta = self.delta.cuda()
            self.weight = self.weight.cuda()
            self.bias = self.bias.cuda()
            self.long_short_pref_weight = self.long_short_pref_weight.cuda()
            self.long_short_pref_bias = self.long_short_pref_bias.cuda()
        self.node_emb.requires_grad = True
        self.delta.requires_grad = True
        self.weight.requires_grad = True
        self.bias.requires_grad = True
        self.long_short_pref_weight.requires_grad = True
        self.long_short_pref_bias.requires_grad = True
        self.opt = Adam(lr=self.lr,
                        params=[self.node_emb, self.delta, self.weight, self.bias, self.long_short_pref_weight,
                                self.long_short_pref_bias], weight_decay=self.decay)
        self.loss = torch.FloatTensor()

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        batch = s_nodes.size()[0]
        s_node_emb = torch.index_select(self.node_emb, 0, s_nodes.view(-1)).view(batch, -1)
        t_node_emb = torch.index_select(self.node_emb, 0, t_nodes.view(-1)).view(batch, -1)
        h_node_emb = torch.index_select(self.node_emb, 0, h_nodes.view(-1)).view(batch, self.hist_len, -1)
        hidden_h_node_emb = torch.relu(torch.matmul(h_node_emb, self.weight) + self.bias)
        attention = softmax((torch.mul(s_node_emb.unsqueeze(1), hidden_h_node_emb).sum(dim=2)), dim=1)
        p_mu = torch.mul(s_node_emb, t_node_emb).sum(dim=1)
        p_alpha = torch.mul(h_node_emb, t_node_emb.unsqueeze(1)).sum(dim=2)
        self.temp_array1 += p_alpha.mean(dim=0).data.cpu().numpy()

        self.delta.data.clamp_(min=1e-6)
        delta = torch.index_select(self.delta, 0, s_nodes.view(-1)).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)

        d_time.data.clamp_(min=1e-6)

        if self.use_user_pref_attention:
            long_short_embedding = torch.cat([s_node_emb, torch.mean(h_node_emb, dim=1)], dim=1)
            pref_hidden = torch.softmax(torch.relu(
                torch.matmul(long_short_embedding, self.long_short_pref_weight) + self.long_short_pref_bias), dim=1)
            self.temp_value3 += pref_hidden[:, 0].mean().data
            self.temp_value4 += pref_hidden[:, 1].mean().data

            long_pref_weight = pref_hidden[:, 0]
            short_pref_weight = pref_hidden[:, 1]
        else:
            long_pref_weight = torch.zeros(batch, dtype=torch.float) + 0.5
            short_pref_weight = torch.zeros(batch, dtype=torch.float) + 0.5
            if torch.cuda.is_available():
                long_pref_weight = long_pref_weight.cuda()
                short_pref_weight = short_pref_weight.cuda()
        self.temp_value1 += long_pref_weight.mean().data
        self.temp_value2 += short_pref_weight.mean().data

        if self.use_hist_attention:
            p_lambda = long_pref_weight * p_mu + short_pref_weight * (
                    attention * p_alpha * torch.exp(torch.neg(delta) * d_time) * h_time_mask).sum(dim=1)

        else:
            p_lambda = long_pref_weight * p_mu + short_pref_weight * (
                    p_alpha * torch.exp(torch.neg(delta) * d_time) * h_time_mask).sum(dim=1)
        n_node_emb = torch.index_select(self.node_emb, 0, n_nodes.view(-1)).view(batch, self.neg_size, -1)
        n_mu = torch.mul(s_node_emb.unsqueeze(1), n_node_emb).sum(dim=2)
        n_alpha = torch.mul(h_node_emb.unsqueeze(2), n_node_emb.unsqueeze(1)).sum(dim=3)
        long_pref_weight = long_pref_weight.unsqueeze(1)
        short_pref_weight = short_pref_weight.unsqueeze(1)
        if self.use_hist_attention:
            n_lambda = long_pref_weight.detach() * n_mu + short_pref_weight.detach() * (
                    attention.detach().unsqueeze(2) * n_alpha * (torch.exp(torch.neg(delta) * d_time).unsqueeze(2)) * (
                h_time_mask.unsqueeze(2))).sum(dim=1)
        else:
            n_lambda = long_pref_weight.detach() * n_mu + short_pref_weight.detach() * (
                    n_alpha * (torch.exp(torch.neg(delta) * d_time).unsqueeze(2)) * (
                h_time_mask.unsqueeze(2))).sum(dim=1)

        self.temp_value5 += p_mu.mean().data
        self.temp_value6 += n_mu.mean().data
        return p_lambda, n_lambda

    def loss_func(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
        loss = -torch.log(torch.sigmoid(p_lambdas.unsqueeze(1)-n_lambdas)).sum(dim=1)
        return loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        self.opt.zero_grad()
        loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
        loss = loss.sum()
        self.loss += loss.data
        loss.backward()
        self.opt.step()

    def train(self):
        self.epoch_temp = 0
        for epoch in range(self.epochs):
            self.epoch_temp = epoch
            self.temp_value1 = 0.0
            self.temp_value2 = 0.0
            self.temp_value3 = 0.0
            self.temp_value4 = 0.0
            self.temp_value5 = 0.0
            self.temp_value6 = 0.0
            self.temp_array1 = np.zeros(self.hist_len)
            self.loss = 0.0

            loader = DataLoader(self.data_tr, batch_size=self.batch, shuffle=True, num_workers=self.num_workers)
            for i_batch, sample_batched in enumerate(loader):

                if torch.cuda.is_available():
                    self.update(sample_batched['source_node'].type(LType).cuda(),
                                sample_batched['target_node'].type(LType).cuda(),
                                sample_batched['target_time'].type(FType).cuda(),
                                sample_batched['neg_nodes'].type(LType).cuda(),
                                sample_batched['history_nodes'].type(LType).cuda(),
                                sample_batched['history_times'].type(FType).cuda(),
                                sample_batched['history_masks'].type(FType).cuda())
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_masks'].type(FType))

            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data_tr)) + '\n')
            sys.stdout.flush()
            if ((epoch + 1) % self.test_and_save_step == 0) or epoch == 0 or epoch == 4 or epoch == 9:
                self.recommend(epoch, is_new_item=False)
                self.recommend(epoch, is_new_item=True)
            print("long_pref_weight.mean(): {}".format(self.temp_value1 / i_batch))
            print("short_pref_weight.mean(): {}".format(self.temp_value2 / i_batch))
            print("long_pref_hidden.mean(): {}".format(self.temp_value3 / i_batch))
            print("short_pref_hidden.mean(): {}".format(self.temp_value4 / i_batch))
            print("alpha.mean(): {}".format(self.temp_array1 / i_batch))
            print("p_mu.mean(): {}".format(self.temp_value5 / i_batch))
            print("n_mu.mean(): {}".format(self.temp_value6 / i_batch))
            print("==========================")

    def recommend(self, epoch, is_new_item=False):
        count_all = 0
        rate_all_sum = 0
        recall_all_sum = np.zeros(self.top_n)
        MRR_all_sum = np.zeros(self.top_n)

        if is_new_item:
            loader = DataLoader(self.data_te_new, batch_size=self.batch, shuffle=False, num_workers=self.num_workers)
        else:
            loader = DataLoader(self.data_te_old, batch_size=self.batch, shuffle=False, num_workers=self.num_workers)
        for i_batch, sample_batched in enumerate(loader):
            if torch.cuda.is_available():
                rate_all, recall_all, MRR_all = \
                    self.evaluate(sample_batched['source_node'].type(LType).cuda(),
                                  sample_batched['target_node'].type(LType).cuda(),
                                  sample_batched['target_time'].type(FType).cuda(),
                                  sample_batched['history_nodes'].type(LType).cuda(),
                                  sample_batched['history_times'].type(FType).cuda(),
                                  sample_batched['history_masks'].type(FType).cuda())
            else:
                rate_all, recall_all, MRR_all = \
                    self.evaluate(sample_batched['source_node'].type(LType),
                                  sample_batched['target_node'].type(LType),
                                  sample_batched['target_time'].type(FType),
                                  sample_batched['history_nodes'].type(LType),
                                  sample_batched['history_times'].type(FType),
                                  sample_batched['history_masks'].type(FType))
            count_all += self.batch
            rate_all_sum += rate_all
            recall_all_sum += recall_all
            MRR_all_sum += MRR_all

        rate_all_sum_avg = rate_all_sum * 1. / count_all
        recall_all_avg = recall_all_sum * 1. / count_all
        MRR_all_avg = MRR_all_sum * 1. / count_all
        if is_new_item:
            logging.info('=========== testing next new item epoch: {} ==========='.format(epoch))
            logging.info('count_all_next_new: {}'.format(count_all))
            logging.info('rate_all_sum_avg_next_new: {}'.format(rate_all_sum_avg))
            logging.info('recall_all_avg_next_new: {}'.format(recall_all_avg))
            logging.info('MRR_all_avg_next_new: {}'.format(MRR_all_avg))
        else:
            logging.info('~~~~~~~~~~~~~ testing next item epoch: {} ~~~~~~~~~~~~~'.format(epoch))
            logging.info('count_all_next: {}'.format(count_all))
            logging.info('rate_all_sum_avg_next: {}'.format(rate_all_sum_avg))
            logging.info('recall_all_avg_next: {}'.format(recall_all_avg))
            logging.info('MRR_all_avg_next: {}'.format(MRR_all_avg))

    def evaluate(self, s_nodes, t_nodes, t_times, h_nodes, h_times, h_time_mask):
        batch = s_nodes.size()[0]
        all_item_index = torch.arange(0, self.item_count)
        if torch.cuda.is_available():
            all_item_index = all_item_index.cuda()
        all_node_emb = torch.index_select(self.node_emb, 0, all_item_index).detach()

        h_node_emb = torch.index_select(self.node_emb, 0, h_nodes.view(-1)).detach().view(batch, self.hist_len, -1)
        p_alpha = torch.matmul(h_node_emb, torch.transpose(all_node_emb, 0, 1))

        self.delta.data.clamp_(min=1e-6)

        d_time = torch.abs(t_times.unsqueeze(1) - h_times)

        delta = torch.index_select(self.delta, 0, s_nodes.view(-1)).detach().unsqueeze(1)
        s_node_emb = torch.index_select(self.node_emb, 0, s_nodes.view(-1)).detach().view(batch, -1)

        hidden_h_node_emb = torch.relu(torch.matmul(h_node_emb, self.weight.detach()) + self.bias.detach())
        attention = softmax((torch.mul(s_node_emb.unsqueeze(1), hidden_h_node_emb).sum(dim=2)), dim=1)
        p_mu = torch.matmul(s_node_emb, torch.transpose(all_node_emb, 0, 1))
        if self.use_user_pref_attention:
            long_short_embedding = torch.cat([s_node_emb, torch.mean(h_node_emb, dim=1)], dim=1)
            pref_hidden = torch.softmax(torch.relu(
                torch.matmul(long_short_embedding,
                             self.long_short_pref_weight.detach()) + self.long_short_pref_bias.detach()), dim=1)
            long_pref_weight = pref_hidden[:, 0]
            short_pref_weight = pref_hidden[:, 1]
        else:
            long_pref_weight = torch.zeros(batch, dtype=torch.float) + 0.5
            short_pref_weight = torch.zeros(batch, dtype=torch.float) + 0.5
            if torch.cuda.is_available():
                long_pref_weight = long_pref_weight.cuda()
                short_pref_weight = short_pref_weight.cuda()
        long_pref_weight = long_pref_weight.unsqueeze(1)
        short_pref_weight = short_pref_weight.unsqueeze(1)
        if self.use_hist_attention:
            p_lambda = long_pref_weight * p_mu + short_pref_weight * (
                    p_alpha * (attention * torch.exp(torch.neg(delta) * d_time) * h_time_mask).unsqueeze(2)).sum(
                dim=1)
        else:
            p_lambda = long_pref_weight * p_mu + short_pref_weight * (
                    p_alpha * (torch.exp(torch.neg(delta) * d_time) * h_time_mask).unsqueeze(2)).sum(dim=1)

        rate_all_sum = 0
        recall_all = np.zeros(self.top_n)
        MRR_all = np.zeros(self.top_n)

        t_nodes_list = t_nodes.cpu().numpy().tolist()
        p_lambda_numpy = p_lambda.cpu().numpy()
        for i in range(len(t_nodes_list)):
            t_node = t_nodes_list[i]
            p_lambda_numpy_i_item = p_lambda_numpy[i]
            prob_index = np.argsort(-p_lambda_numpy_i_item).tolist()
            gnd_rate = prob_index.index(t_node) + 1
            rate_all_sum += gnd_rate
            if gnd_rate <= self.top_n:
                recall_all[gnd_rate - 1:] += 1
                MRR_all[gnd_rate - 1:] += 1. / gnd_rate
        return rate_all_sum, recall_all, MRR_all

    def save_parameter_value(self, path, parameter, data_type):
        if torch.cuda.is_available():
            parameter_cpu = parameter.cpu().data.numpy()
        else:
            parameter_cpu = parameter.data.numpy()
        writer = open(path, 'w')
        if data_type == "vector":
            writer.write('%d\n' % (parameter_cpu.shape[0]))
            writer.write('\t'.join(str(d) for d in parameter_cpu))
        elif data_type == "matrix":
            dim_0, dim_1 = parameter_cpu.shape
            writer.write('%d\t%d\n' % (dim_0, dim_1))
            for n_idx in range(dim_0):
                writer.write('\t'.join(str(d) for d in parameter_cpu[n_idx]) + '\n')
        else:
            pass
        writer.close()