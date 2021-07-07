import pymysql
import random
import csv


def child_company_process():
    data_list = []
    with open('E:\\文档\\data\\project_data\\\company_knowledge_base\\graph_init_data\\company2child.txt', 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            if not line:
                continue
            if line not in data_list:
                data_list.append(line)
    with open('company_child_triple.txt', 'w', encoding='utf-8') as fw:
        for l in data_list:
            company_list = l.split('\t')
            if len(company_list) != 2:
                print(l)
                continue
            fw.write(company_list[0] + '\t相关企业\t' + company_list[1] + '\n')


def company_property_clean():
    company_dict = {}
    fw = open('company_entity.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(fw)
    ubaike_title = ['实体名称', 'company_sql_id', '工商注册号', 'social_id', 'trade_type', '主营地区', '企业类型', 'area_code']
    hy88_title_list = ['实体名称', 'company_sql_id', '主营范围', 'area_code', '注册资本', '企业类型', '法人', '描述',
                        '主营地区', '公司邮编', '联系电话', '网站地址', '电子邮件', '公司地址', 'social_id', 'trade_type']
    csv_writer.writerow(hy88_title_list[:-1])
    with open("C:\\Users\\zaa\\Desktop\\hy88\\company_entity.csv", "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i == 0:
                i += 1
                continue
            if row[0] not in company_dict.keys():
                company_dict[row[0]] = {}
                for _index in range(1, len(hy88_title_list)-1):
                    if hy88_title_list[_index] in ['trade_type', 'company_sql_id']:
                        continue
                    company_dict[row[0]][hy88_title_list[_index]] = row[_index]
    with open("C:\\Users\\zaa\\Desktop\\hy88\\ubaike_company_entity.csv", "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i == 0:
                i += 1
                continue
            if row[0] not in company_dict.keys():
                company_dict[row[0]] = {}
                for _index in range(1, len(ubaike_title)):
                    if ubaike_title[_index] in ['trade_type', 'company_sql_id']:
                        continue
                    company_dict[row[0]][ubaike_title[_index]] = row[_index]
    print(len(company_dict))
    for key, value_dict in company_dict.items():
        data_list = [key]
        for title in hy88_title_list[1:-1]:
            if title not in value_dict.keys():
                data_list.append('')
            else:
                data_list.append(value_dict[title])
        csv_writer.writerow(data_list)


def get_sx_train_data():
    """
    将原始训练数据格式转化为标准训练数据集
    :return:
    """
    tag_list = []
    fw_train = open('E:\\PycharmProjects\\knowledge_base\\BiLSTM_model\\dishonesty_data\\train_data.txt', 'w', encoding='utf-8')
    fw_test = open('E:\\PycharmProjects\\knowledge_base\\BiLSTM_model\\dishonesty_data\\test_data.txt', 'w', encoding='utf-8')
    with open('E:\\文档\\data\\project_data\\company_knowledge_base\\kg_train_已清洗.txt', 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip('\t').strip('\n')
            if not line:
                continue
            if random.random() <= 0.98:
                fw = fw_train
            else:
                fw = fw_test
            data_list = line.split('\t')
            for _words_tag in data_list:
                if 'defendant' in _words_tag or 'plaintiff' in _words_tag:
                    word = _words_tag.split('/')[0]
                    tag = _words_tag.split('/')[1]
                    if tag not in tag_list:
                        tag_list.append(tag)
                    fw.write(word[0] + '\t' + 'B-' + tag + '\n')
                    for w in range(1, len(word)-1):
                        fw.write(word[w] + '\t' + 'M-' + tag + '\n')
                    fw.write(word[-1] + '\t' + 'E-' + tag + '\n')
                else:
                    for w in _words_tag:
                        fw.write(w + '\t' + 'O' + '\n')
            fw.write('\n')
    print(tag_list)


def get_scsg_company_data():
    conn = pymysql.connect(host='192.168.130.220', user='showclear', password='showclear', db='DB_SC_DATA', charset='utf8')
    cursor = pymysql.cursors.SSCursor(conn)
    sql = "SELECT GROUP_CONCAT(T1.start_idx), GROUP_CONCAT(T1.end_idx), T2.content FROM label AS T1 INNER JOIN sentence" \
          " AS T2 ON T1.s_id=T2.id WHERE T1.content='企业名称' GROUP BY T1.s_id"
    cursor.execute(sql)
    while True:
        row = cursor.fetchone()
        if not row:
            break
        if ',' in row[0]:
            company_list = []
            sentence = row[2].replace('\t', ' ')
            star_idx_list = row[0].split(',')
            end_idx_list = row[1].split(',')
            for idx_id in range(len(star_idx_list)):
                company = sentence[int(star_idx_list[idx_id]): int(end_idx_list[idx_id])]
                if company not in company_list:
                    company_list.append(company)
            if len(company_list) > 1:
                for c in company_list:
                    print(c)
                print(sentence)
                print('*******************************************')


def get_company_name_train_data():
    tag_list = []
    fw_train = open('E:\\PycharmProjects\\knowledge_base\\BiLSTM_model\\company_name_data\\train_data.txt', 'w', encoding='utf-8')
    fw_test = open('E:\\PycharmProjects\\knowledge_base\\BiLSTM_model\\company_name_data\\test_data.txt', 'a', encoding='utf-8')
    with open('E:\\文档\\data\\project_data\\company_knowledge_base\\company_name_train_clean.txt', 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            if not line:
                continue
            if random.random() <= 0.99:
                fw = fw_train
            else:
                fw = fw_test
            data_list = line.split('\t')
            for words_tag in data_list:
                word = words_tag.split('/')[0]
                tag = words_tag.split('/')[1]
                if tag not in tag_list:
                    tag_list.append(tag)
                if len(word) > 1:
                    fw.write(word[0] + '\t' + 'B-' + tag + '\n')
                    for w in range(1, len(word) - 1):
                        fw.write(word[w] + '\t' + 'M-' + tag + '\n')
                    fw.write(word[-1] + '\t' + 'E-' + tag + '\n')
                else:
                    fw.write(word + '\t' + 'S-' + tag + '\n')
            fw.write('\n')
    print(tag_list)
    fw.close()


def delete_duplicated_data():
    name_list = []
    with open('E:\\PycharmProjects\\knowledge_base\\company_kg\\company_name_ner.txt', 'r', encoding='utf-8') as fr:
        with open('testtest.txt', 'w', encoding='utf-8') as fw:
            for line in fr.readlines():
                line = line.strip('\n')
                if not line:
                    continue
                company_name = line.split('\t')[1]
                if company_name not in name_list:
                    fw.write(line + '\n')
                    name_list.append(company_name)


if __name__ == '__main__':
    # pass
    get_company_name_train_data()
    # get_sx_train_data()
