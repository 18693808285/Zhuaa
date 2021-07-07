# coding=utf-8
import tensorflow as tf
import os
import logging

from BiLSTM_model.model import BiLSTM_CRF
from BiLSTM_model.ner_model_config import get_args_parser
from BiLSTM_model.data import read_dictionary, tag2label_mapping, random_embedding


class CRFPredict:

    def __init__(self, dataset_name, model_name):
        self.config = tf.ConfigProto()
        self.args = get_args_parser(dataset_name, model_name=model_name)
        # self.word2id = read_dictionary(os.path.join('word2id.pkl'))
        self.word2id = read_dictionary(self.args.word2id)
        self.tag2label = tag2label_mapping[self.args.dataset_name]
        self.embeddings = random_embedding(self.word2id, self.args.embedding_dim)
        self.model = None
        self.graph = tf.get_default_graph()
        self.sess = tf.Session(config=self.config)
        paths = {}
        model_path = os.path.join(self.args.output_path, "checkpoints/")
        self.ckpt_file = tf.train.latest_checkpoint(model_path)
        paths['model_path'] = self.ckpt_file
        with self.graph.as_default():
            self.model = BiLSTM_CRF(self.args, self.embeddings, self.tag2label, self.word2id, paths, config=self.config)
            self.model.build_graph()
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.ckpt_file)

    def predict(self, sentence):
        sentence = sentence.strip('\n').strip(' ').strip('\t')
        if not sentence:
            return []
        with self.graph.as_default():
            demo_sent = list(sentence.strip())
            demo_data = [(demo_sent, ['O'] * len(demo_sent))]
            tags = self.model.demo_one(self.sess, demo_data)
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
