import os
import json
import torch
import pickle
import numpy as np
import torch.utils.data as data
from collections import Counter


class Dataset(data.Dataset):
    def __init__(self, split, logger, opt, training=True, pn_mode=None):
        super().__init__()
        self.split = split
        self.max_length = opt['max_length']
        self.max_pos_length = opt['max_pos_length']
        self.processed_data_dir = opt['processed_data_dir']
        self.rel2id = json.load(open(os.path.join(opt['root'], opt['rel2id'])))
        self.word_vec_dir, word2id = self.init_word(os.path.join(opt['root'], opt['vec_dir']))
        self.type2id = json.load(open(os.path.join(opt['root'], opt['type2id'])))
        self.graph = json.load(open(os.path.join(opt['root'], opt['graph'])))
        self.logger = logger
        self.training = training
        self.pn_mode = pn_mode

        if not os.path.exists(opt['trained_models_dir']):  # trained model path
            os.mkdir(opt['trained_models_dir'])

        self.logger.info("Trying to load processed data")

        # self.bag = Bag()
        self.data = pickle.load(
            open(os.path.join(self.processed_data_dir, self.split + '.pkl'), 'rb'))
        self.logger.debug("Load successfully")
        if not training:
            if pn_mode is not None:
                logger.info("Preparing dataset loader for Test in mode %s..." % pn_mode.capitalize())
                self.data = [d for d in self.data if len(d.word) > 1]
            else:
                logger.info("Preparing dataset loader for Test ...")
                self.rel_100, self.rel_200 = self.long_tail(os.path.join(opt['root'], opt['train']))
        else:
            logger.info("Preparing dataset loader for Training...")
        self.rel_num = len(self.rel2id)
        self.type_num = len(self.type2id)
        self.batch_num = np.ceil(len(self.data) / opt['batch_size'])  # np.ceil()向上取整
        self.edge, self.constraint, self.edge_r, self.edge_t = self.build_graph()
        self.logger.info(
            "total bag nums: %d\n" % (self.__len__()))  # data 数据按包（（实体1，实体2，关系类型）为一个包。关系类型53种，共有293162个包）进行存储

    def init_word(self, vec_dir):
        if not os.path.isdir(self.processed_data_dir):
            os.mkdir(self.processed_data_dir)
        word2id_dir = os.path.join(self.processed_data_dir, 'word2id.json')
        word_vec_dir = os.path.join(self.processed_data_dir, 'word_vec.npy')
        try:
            word2id = json.load(open(word2id_dir))
        except FileNotFoundError:
            f = open(vec_dir, encoding='utf-8')
            num, dim = [int(x) for x in f.readline().split()[:2]]
            word2id = {}
            word_vec = np.zeros([num + 2, dim], dtype=np.float32)
            for line in f.readlines():
                line = line.strip().split()
                word_vec[len(word2id)] = np.array(line[1:])
                word2id[line[0].lower()] = len(word2id)
            f.close()
            word_vec[len(word2id)] = np.random.randn(dim) / np.sqrt(dim)
            word2id['[UNK]'] = len(word2id)
            word2id['[PAD]'] = len(word2id)
            np.save(word_vec_dir, word_vec)
            json.dump(word2id, open(word2id_dir, 'w'))
        return word_vec_dir, word2id

    def long_tail(self, train_dir):
        try:
            self.logger.debug("Trying to load long tail data...")
            long_tail = json.load(open(os.path.join(self.processed_data_dir, 'long_tail.json')))
        except FileNotFoundError:
            self.logger.debug("Long tail data does not exist, extracting...")
            train_data = json.load(open(train_dir))
            rel_ins = [self.rel2id[ins['rel']] for ins in train_data if
                       (ins['rel'] != 'NA' and ins['rel'] in self.rel2id)]
            stat = Counter(rel_ins)
            rel_100 = [k - 1 for k, v in stat.items() if v < 100]  # k-1 because 'NA' is removed
            rel_200 = [k - 1 for k, v in stat.items() if v < 200]
            long_tail = {"rel_100": rel_100, "rel_200": rel_200}
            json.dump(long_tail, open(os.path.join(self.processed_data_dir, 'long_tail.json'), 'w'))
        return long_tail["rel_100"], long_tail["rel_200"]

    def build_graph(self):
        self.logger.debug("Building Graph...")
        sub = []   # bipartite constraint graph
        obj = []

        sub_r = []  # relations dependent graph
        obj_r = []
        sub_t = []  # entities dependent graph
        obj_t = []

        constraint = [[[], []] for _ in range(self.rel_num)]
        # type_rels = [[] for _ in range(self.type_num)]
        for rel, types in self.graph.items():
            r = self.rel2id[rel]
            r_node = r + self.type_num  # rel id -> node id
            for type in types:
                h, t = self.type2id[type[0]], self.type2id[type[1]]
                # h -> r -> t
                sub += [h, r_node]
                obj += [r_node, t]

                sub_t.append(h)
                obj_t.append(t)

                constraint[r][0].append(h)
                constraint[r][1].append(t)
            constraint[r] = torch.tensor(constraint[r], dtype=torch.long)
        # for NA
        for i in range(self.type_num):
            sub += [i, self.type_num]
            obj += [self.type_num, i]

            sub_t.append(i)
            obj_t.append(i)

            constraint[0][0].append(i)
            constraint[0][1].append(i)
        for i in range(len(sub)):
            if sub[i] < self.type_num:
                t_i = sub[i]
                r_i = obj[i]
            else:
                t_i = obj[i]
                r_i = sub[i]
            for j in range(i + 1, len(sub)):
                if sub[j] < self.type_num:
                    t_j = sub[j]
                    r_j = obj[j]
                else:
                    t_j = obj[j]
                    r_j = sub[j]
                if t_i == t_j:
                    sub_r.append(r_i - self.type_num)  # BACK TO r index
                    obj_r.append(r_j - self.type_num)

        constraint[0] = torch.tensor(constraint[0], dtype=torch.long)
        edge = torch.tensor([sub, obj], dtype=torch.long)
        edge_t = torch.tensor([sub_t, obj_t], dtype=torch.long)
        edge_r = torch.tensor([sub_r, obj_r], dtype=torch.long)

        if torch.cuda.is_available():
            edge = edge.cuda()
            edge_t = edge_t.cuda()
            edge_r = edge_r.cuda()
            for i in range(self.rel_num):
                constraint[i] = constraint[i].cuda()
        self.logger.debug("Finish building")
        return edge, constraint, edge_r, edge_t

    def rel_weight(self):
        self.logger.debug("Calculating the n-class weight")
        rel_ins = []
        for bag in self.data:
            rel_ins.extend(bag.rel)
        stat = Counter(rel_ins)
        class_weight = torch.ones(self.rel_num, dtype=torch.float32)
        for k, v in stat.items():
            class_weight[k] = 1. / v ** 0.05
        if torch.cuda.is_available():
            class_weight = class_weight.cuda()
        return class_weight

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        bag = self.data[index]
        sen_num = len(bag.word)
        select = torch.arange(sen_num)
        if not self.training and self.pn_mode is not None:
            if self.pn_mode == 'one' and sen_num > 1:
                select = torch.tensor(np.random.choice(sen_num, 1), dtype=torch.long)
            elif self.pn_mode == 'two' and sen_num > 2:
                select = torch.tensor(np.random.choice(sen_num, 2, replace=False), dtype=torch.long)
        word = torch.tensor(bag.word, dtype=torch.long)[select]
        pos1 = torch.tensor(bag.pos1, dtype=torch.long)[select]
        pos2 = torch.tensor(bag.pos2, dtype=torch.long)[select]
        ent1 = torch.tensor(bag.ent1, dtype=torch.long)[select]
        ent2 = torch.tensor(bag.ent2, dtype=torch.long)[select]
        mask = torch.tensor(bag.mask, dtype=torch.long)[select]
        type = torch.tensor(bag.type, dtype=torch.long)[select]
        if self.training:
            rel = torch.tensor(bag.rel[0], dtype=torch.long)
        else:
            rel = torch.zeros(len(self.rel2id), dtype=torch.long)
            for i in set(bag.rel):
                rel[i] = 1
        return word, pos1, pos2, ent1, ent2, mask, type, rel


def collate_fn(X):
    X = list(zip(*X))
    word, pos1, pos2, ent1, ent2, mask, type, rel = X
    scope = []
    ind = 0
    for w in word:
        scope.append((ind, ind + len(w)))
        ind += len(w)
    scope = torch.tensor(scope, dtype=torch.long)
    word = torch.cat(word, 0)
    pos1 = torch.cat(pos1, 0)
    pos2 = torch.cat(pos2, 0)
    ent1 = torch.cat(ent1, 0)
    ent2 = torch.cat(ent2, 0)
    mask = torch.cat(mask, 0)
    type = torch.cat(type, 0)
    rel = torch.stack(rel)
    return word, pos1, pos2, ent1, ent2, mask, type, scope, rel


def data_loader(split, logger, opt, shuffle, training=True, pn_mode=None, num_workers=0):
    dataset = Dataset(split, logger, opt, training, pn_mode)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=opt['batch_size'],
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return loader
