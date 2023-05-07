import numpy as np
import json
import random
from transformers import BertTokenizer


class data_sampler(object):

    def __init__(self, config=None, seed=None):

        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])

        self.id2rel, self.rel2id = self._read_relations(config.relation_file)
        self.id2sent = {}
        self.training_data = self.load_data(config.training_file)
        self.valid_data = self.load_data(config.valid_file)
        self.test_data = self.load_data(config.test_file)

        self.task_length = config.task_length
        rel_index = np.load(config.rel_index)
        rel_cluster_label = np.load(config.rel_cluster_label)
        self.cluster_to_labels = {}
        for index, i in enumerate(rel_index):
            if rel_cluster_label[index] in self.cluster_to_labels.keys():
                self.cluster_to_labels[rel_cluster_label[index]].append(i-1)
            else:
                self.cluster_to_labels[rel_cluster_label[index]] = [i-1]

        self.seed = seed
        if self.seed != None:
            self.set_seed(self.seed)

        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)

        self.batch = 0

        # record relations
        self.seen_relations = []
        self.history_test_data = {}

    def load_data(self, file):
        samples = []
        with open(file) as file_in:
            for line in file_in:
                items = line.strip().split('\t')
                if (len(items[0]) > 0):
                    relation_ix = int(items[0])
                    if items[1] != 'noNegativeAnswer':
                        candidate_ixs = [int(ix) for ix in items[1].split()]
                        sentence = items[2].split('\n')[0]
                        headent = items[3]
                        headidx = [int(ix) for ix in items[4].split()]
                        tailent = items[5]
                        tailidx = [int(ix) for ix in items[6].split()]
                        headid = items[7]
                        tailid = items[8]
                        samples.append(
                            [relation_ix, candidate_ixs, sentence, headent, headidx, tailent, tailidx,
                             headid, tailid])
        read_data = [[] for i in range(self.config.num_of_relation)]

        for sample in samples:
            text = sample[2]
            split_text = text.split(" ")
            new_headent = ' [E11] ' + sample[3] + ' [E12] '
            new_tailent = ' [E21] ' + sample[5] + ' [E22] '
            if sample[4][0] < sample[6][0]:
                new_text = " ".join(split_text[0:sample[4][0]]) + new_headent + " ".join(
                    split_text[sample[4][-1] + 1:sample[6][0]]) \
                           + new_tailent + " ".join(split_text[sample[6][-1] + 1:len(split_text)])
            else:
                new_text = " ".join(split_text[0:sample[6][0]]) + new_tailent + " ".join(
                    split_text[sample[6][-1] + 1:sample[4][0]]) \
                           + new_headent + " ".join(split_text[sample[4][-1] + 1:len(split_text)])

            tokenized_sample = {}
            tokenized_sample['relation'] = sample[0] - 1
            tokenized_sample['neg_labels'] = [can_idx - 1 for can_idx in sample[1]]
            tokenized_sample['tokens'] = self.tokenizer.encode(new_text,
                                                               padding='max_length',
                                                               truncation=True,
                                                               max_length=self.config.max_length)
            self.id2sent[len(self.id2sent)] = tokenized_sample['tokens']
            read_data[tokenized_sample['relation']].append(tokenized_sample)
        return read_data

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)

        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)
        print(self.shuffle_index)


    def _read_relations(self, file):
        '''
        :param file: input relation file
        :return:  a list of relations, and a mapping from relations to their ids.
        '''

        id2rel = []
        rel2id = {}
        with open(file) as file_in:
            for line in file_in:
                id2rel.append(line.strip())
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch == self.task_length:
            self.batch = 0
            raise StopIteration()

        indexs = self.cluster_to_labels[self.shuffle_index[self.batch]]  # 每个任务出现的id
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indexs:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])

            cur_training_data[self.id2rel[index]] = self.training_data[index]
            cur_valid_data[self.id2rel[index]] = self.valid_data[index]
            cur_test_data[self.id2rel[index]] = self.test_data[index]
            self.history_test_data[self.id2rel[index]] = self.test_data[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations

    def get_id2sent(self):
        return self.id2sent