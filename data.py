# coding=utf-8
import os
import pickle
import random
import codecs
import numpy as np

tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


tag2label_plan_digitization = {"O": 0, "B-nt": 1, "M-nt": 2, "E-nt": 3, "B-death": 4, "M-death": 5, "E-death": 6,
                               "B-heart": 7,  "M-heart": 8, "E-heart": 9, "B-ecLoss": 10, "M-ecLoss": 11, "E-ecLoss": 12,
                               "B-duration": 13, "M-duration": 14, "E-duration": 15, "B-quakeStreng": 16,
                               "M-quakeStreng": 17, "E-quakeStreng": 18, "B-aqi": 19, "M-aqi": 20, "E-aqi": 21,
                               "B-m": 22, "M-m": 23, "E-m": 24, "S-m": 25,  "B-lifeSecurity": 26, "M-lifeSecurity": 27,
                               "E-lifeSecurity": 28, "B-quantifier": 29, "M-quantifier": 30, "E-quantifier": 31,
                               "S-quantifier": 32, "B-keywords": 33, "M-keywords": 34, "E-keywords": 35,
                               "B-emergTrans": 36, "M-emergTrans": 37, "E-emergTrans": 38, 'B-casualties': 39, 'E-casualties': 40
                        }


tag2label_plan_digitization_v2 = {"O": 0, "B-nt": 1, "M-nt": 2, "E-nt": 3, "B-death": 4, "M-death": 5, "E-death": 6,
                                  "B-heart": 7,  "M-heart": 8, "E-heart": 9, "B-ecLoss": 10, "M-ecLoss": 11, "E-ecLoss": 12,
                                  "B-duration": 13, "M-duration": 14, "E-duration": 15, "B-quakeStreng": 16,
                                  "M-quakeStreng": 17, "E-quakeStreng": 18, "B-aqi": 19, "M-aqi": 20, "E-aqi": 21,
                                  "B-m": 22, "M-m": 23, "E-m": 24, 'S-m': 25,  'B-lifeSecurity': 26, 'M-lifeSecurity': 27,
                                  "E-lifeSecurity": 28, "B-quantifier": 29, "M-quantifier": 30, "E-quantifier": 31,
                                  "S-quantifier": 32, "B-keywords": 33, "M-keywords": 34, "E-keywords": 35,
                                  "B-emergTrans": 36, "M-emergTrans": 37, "E-emergTrans": 38, 'B-casualties': 39,
                                  'E-casualties': 40,  "B-org": 41, "M-org": 42, "E-org": 43
                        }

tag2label_dishonesty = {"O": 0, "B-defendant": 1, "M-defendant": 2, "E-defendant": 3, "B-plaintiff": 4,
                        "M-plaintiff": 5, "E-plaintiff": 6}

tag2label_company_name = {"S-tail": 0, "B-ns": 1, "M-ns": 2, "E-ns": 3, "B-nz": 4, "M-nz": 5, "E-nz": 6,
                          "B-trade": 7, "M-trade": 8, "E-trade": 9, "B-tail": 10, "M-tail": 11, "E-tail": 12,
                          "B-type": 13, "M-type": 14, "E-type": 15, "O": 16, "S-trade": 17, "S-nz": 18, "S-ns": 19}

tag2label_mapping = {
                        'plan_digitization_data': tag2label_plan_digitization,
                        'plan_digitization_v2_data': tag2label_plan_digitization_v2,
                        'dishonesty_data': tag2label_dishonesty,
                        'company_name_data': tag2label_company_name

                    }


def build_character_embeddings(pretrained_emb_path, embeddings_path, word2id, embedding_dim):
    print('loading pretrained embeddings from {}'.format(pretrained_emb_path))
    pre_emb = {}
    for line in codecs.open(pretrained_emb_path, 'r', 'utf-8'):
        line = line.strip().split()
        if len(line) == embedding_dim + 1:
            pre_emb[line[0]] = [float(x) for x in line[1:]]
    word_ids = sorted(word2id.items(), key=lambda x: x[1])
    characters = [c[0] for c in word_ids]
    embeddings = list()
    for i, ch in enumerate(characters):
        if ch in pre_emb:
            embeddings.append(pre_emb[ch])
        else:
            embeddings.append(np.random.uniform(-0.25, 0.25, embedding_dim).tolist())
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(embeddings_path, embeddings)


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip('\n').split('\t')
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count=1):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    # 返回一个序列中长度最长的那条样本的长度
    max_len = max(map(lambda x: len(x), sequences))

    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []
        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


if __name__ == '__main__':
    pass
