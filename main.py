# coding=utf-8
import tensorflow as tf
import numpy as np
import os, argparse, time, random
from BiLSTM_model.model import BiLSTM_CRF
from BiLSTM_model.utils import str2bool, get_logger, get_entity
from BiLSTM_model.data import read_corpus, read_dictionary, tag2label_mapping, random_embedding, vocab_build, \
    build_character_embeddings

# Session configuration
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.3  # need ~700MB GPU memory

# hyper parameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--dataset_name', type=str, default='company_name_data',
                    help='choose a dataset(dishonesty_data, plan_digitization_data, company_name_data)')
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
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1594797663', help='model for test and demo')
args = parser.parse_args()

# vocabulary build
if not os.path.exists(os.path.join(args.dataset_name, 'word2id.pkl')):
    vocab_build(os.path.join(args.dataset_name, 'word2id.pkl'),
                os.path.join(args.dataset_name, 'train_data.txt'))

# get word dictionary
word2id = read_dictionary(os.path.join(args.dataset_name, 'word2id.pkl'))

# build char embeddings
if not args.use_pre_emb:
    embeddings = random_embedding(word2id, args.embedding_dim)
    log_pre = 'not_use_pretrained_embeddings'
else:
    pre_emb_path = os.path.join('.', args.pretrained_emb_path)
    embeddings_path = os.path.join('data_path', args.dataset_name, 'pretrain_embedding.npy')
    if not os.path.exists(embeddings_path):
        build_character_embeddings(pre_emb_path, embeddings_path, word2id, args.embedding_dim)
    embeddings = np.array(np.load(embeddings_path), dtype='float32')
    log_pre = 'use_pretrained_embeddings'

# choose tag2label
tag2label = tag2label_mapping[args.dataset_name]

# read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join(args.dataset_name, 'train_data.txt')
    test_path = os.path.join(args.dataset_name, 'test_data.txt')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    test_size = len(test_data)

# paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('model_path', args.dataset_name, timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, args.dataset_name + log_pre + "_log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


def train_model():
    # training model
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    # train model on the whole training data
    print("train data: {}".format(len(train_data)))
    print("test data: {}".format(test_size))
    model.train(train=train_data, dev=test_data)  # use test_data.txt as the dev_data to see overfitting phenomena


def test_model():
    # testing model
    ckpt_file = tf.train.latest_checkpoint(model_path)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    model.test(test_data)


def predict(line):
    ckpt_file = tf.train.latest_checkpoint(model_path)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt_file)
        demo_sent = list(line.strip())
        demo_data = [(demo_sent, ['O'] * len(demo_sent))]
        tag = model.demo_one(sess, demo_data)
    print(line)
    print(tag)


if __name__ == '__main__':
    # pass
    train_model()
