# coding=utf-8
import tensorflow as tf
import os
import logging

from BiLSTM_model.model import BiLSTM_CRF
from BiLSTM_model.ner_model_config import get_args_parser
from BiLSTM_model.data import read_dictionary, tag2label_mapping, random_embedding


class CRFPredict:

    def __init__(self, dataset_list, model_list):
        self.config = tf.ConfigProto()
        self.init_dishonesty_model(dataset_list[1], model_list[0])
        self.init_company_model(dataset_list[0], model_list[0])

    def init_company_model(self, dataset_name, model_name):
        model_args = get_args_parser(dataset_name, model_name=model_name)
        tag2label = tag2label_mapping[model_args.dataset_name]
        word2id = read_dictionary('E:\\PycharmProjects\\knowledge_base\\BiLSTM_model\\company_name_data\\word2id.pkl')
        embeddings = random_embedding(word2id, 300)
        self.g1 = tf.Graph()
        self.sess1 = tf.Session(config=self.config, graph=self.g1)
        self.model1 = None
        paths = {}
        model_path = os.path.join(model_args.output_path, "checkpoints/")
        ckpt_file = tf.train.latest_checkpoint(model_path)
        paths['model_path'] = ckpt_file
        with self.sess1.as_default():
            with self.g1.as_default():
                self.model1 = BiLSTM_CRF(model_args, embeddings, tag2label, word2id, paths, config=self.config)
                self.model1.build_graph()
                saver = tf.train.Saver()
                saver.restore(self.sess1, ckpt_file)

    def init_dishonesty_model(self, dataset_name, model_name):
        model_args = get_args_parser(dataset_name, model_name=model_name)
        tag2label = tag2label_mapping[model_args.dataset_name]
        word2id = read_dictionary('E:\\PycharmProjects\\knowledge_base\\BiLSTM_model\\dishonesty_data\\word2id.pkl')
        embeddings = random_embedding(word2id, 300)
        self.g2 = tf.get_default_graph()
        self.sess2 = tf.Session(config=self.config, graph=self.g2)
        self.model2 = None
        paths = {}
        model_path = os.path.join(model_args.output_path, "checkpoints/")
        ckpt_file = tf.train.latest_checkpoint(model_path)
        paths['model_path'] = ckpt_file
        with self.sess2.as_default():
            with self.g2.as_default():
                self.model2 = BiLSTM_CRF(model_args, embeddings, tag2label, word2id, paths, config=self.config)
                self.model2.build_graph()
                saver = tf.train.Saver()
                saver.restore(self.sess2, ckpt_file)

    def company_predict(self, sentence):
        sentence = sentence.strip('\n').strip(' ').strip('\t')
        if not sentence:
            return []
        with self.g1.as_default():
            demo_sent = list(sentence.strip())
            demo_data = [(demo_sent, ['O'] * len(demo_sent))]
            tags = self.model1.demo_one(self.sess1, demo_data)
        ner_list = self.get_ner(demo_sent, tags)
        return ner_list

    def dishonesty_predict(self, sentence):
        sentence = sentence.strip('\n').strip(' ').strip('\t')
        if not sentence:
            return []
        with self.g2.as_default():
            demo_sent = list(sentence.strip())
            demo_data = [(demo_sent, ['O'] * len(demo_sent))]
            tags = self.model2.demo_one(self.sess2, demo_data)
        ner_list = self.get_ner(demo_sent, tags)
        return ner_list

    def get_ner(self, sentence, tag_list):
        ner_list = []
        if len(sentence) != len(tag_list):
            logging.warning('--ner error, tag len error')
            print('tag len error')
            return None
        ner_text = ''
        ner_tag = ''
        for char_id in range(len(sentence)):
            if tag_list[char_id] == 'O':
                if ner_text:
                    ner_list.append(ner_text + '/' + ner_tag)
                    ner_text = ''
                    ner_tag = ''
                ner_list.append(sentence[char_id] + '/O')
            elif tag_list[char_id][0] == 'B':
                if ner_text:
                    if len(ner_text) == 1 and tag_list[char_id].split('-')[-1] == ner_tag:
                        ner_text += sentence[char_id]
                        continue
                    ner_list.append(ner_text + '/' + ner_tag)
                ner_text = sentence[char_id]
                ner_tag = tag_list[char_id].split('-')[-1]
            elif tag_list[char_id][0] == 'S':
                if ner_text:
                    if tag_list[char_id].split('-')[-1] == ner_tag:
                        ner_text += sentence[char_id]
                        continue
                    ner_list.append(ner_text + '/' + ner_tag)
                    ner_text = ''
                    ner_tag = ''
                ner_list.append(sentence[char_id] + '/' + tag_list[char_id][2:])
            elif tag_list[char_id][0] == 'M':
                if not ner_tag:
                    ner_tag = tag_list[char_id].split('-')[-1]
                ner_text += sentence[char_id]
            elif tag_list[char_id][0] == 'E':
                if not ner_tag:
                    ner_tag = tag_list[char_id].split('-')[-1]
                ner_text += sentence[char_id]
                ner_list.append(ner_text + '/' + ner_tag)
                ner_text = ''
                ner_tag = ''
        if ner_text and ner_tag:
            ner_list.append(ner_text + '/' + ner_tag)
        return ner_list


if __name__ == '__main__':
    pass
