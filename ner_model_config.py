# -*- coding: utf-8 -*-

import argparse
import os
from BiLSTM_model.utils import str2bool


def get_args_parser(dataset_name='plan_digitization_data', model_name='1598594847'):
    root_path = r'E:\PycharmProjects\knowledge_base\BiLSTM_model'
    parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
    parser.add_argument('--dataset_name', type=str, default=dataset_name,
                        help='choose a dataset(MSRA, ResumeNER, WeiboNER,人民日报, plan_digitization_data)')
    parser.add_argument('--batch_size', type=int, default=4, help='#sample of each minibatch')
    parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
    parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
    parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
    parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
    parser.add_argument('--use_pre_emb', type=str2bool, default=False,
                        help='use pre_trained char embedding or init it randomly')
    parser.add_argument('--pretrained_emb_path', type=str, default='sgns.wiki.char', help='pretrained embedding path')
    parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
    # parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
    parser.add_argument('--output_path', type=str, default=os.path.join(root_path, 'model_path', dataset_name, model_name), help='model for test and demo')
    parser.add_argument('--word2id', type=str, default=os.path.join(root_path, dataset_name, 'word2id.pkl'), help='model for test and demo')
    return parser.parse_args()

