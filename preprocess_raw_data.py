import json
from tqdm import tqdm
import pickle
import numpy as np
import os
from config import config
from utils import AverageMeter, setup_seed, get_logger


class Bag:
    def __init__(self):
        self.word = []
        self.pos1 = []
        self.pos2 = []
        self.ent1 = []
        self.ent2 = []
        self.mask = []
        self.length = []
        self.type = []
        self.rel = []

def preprocess(split,opt):

    rel2id = json.load(open("data/rel2id.json"))

    type2id = json.load(open("data/type2id.json"))

    try:
        word2id = json.load(open("processed_data/word2id.json"))
    except FileNotFoundError:
        f = open('data/vec.txt', encoding='utf-8')
        num, dim = [int(x) for x in f.readline().split()[:2]]
        word2id = {}
        word_vec = np.zeros([num + 2, dim], dtype=np.float32)
        for line in f.readlines():
            line = line.strip().split()
            word_vec[len(word2id)] = np.array(line[1:])
            word2id[line[0]] = len(word2id)
        f.close()
        word_vec[len(word2id)] = np.random.randn(dim) / np.sqrt(dim)
        word2id['[UNK]'] = len(word2id)
        word2id['[PAD]'] = len(word2id)
        np.save("processed_data/word_vec.npy", word_vec)
        json.dump(word2id, open("processed_data/word2id.json", 'w'))
        word2id = json.load(open("processed_data/word2id.json"))



    # self.logger.debug("Processed data does not exist, preprocessing...")
    # Load files
    ori_data = json.load(open('data/'+split+'.json'))
    # Sort data by entities and relations
    ori_data.sort(key=lambda a: a['sub']['id'] + '#' + a['obj']['id'] + '#' + a['rel'])
    # Pre-process data
    last_bag = None
    data = []
    bag = Bag()
    for ins in tqdm(ori_data):
        _rel = rel2id[ins['rel']] if ins['rel'] in rel2id else rel2id['NA']
        if split=='train':
            cur_bag = (ins['sub']['id'], ins['obj']['id'], str(_rel))  # used for train
        else:
            cur_bag = (ins['sub']['id'], ins['obj']['id'])  # used for test

        if cur_bag != last_bag:
            if last_bag is not None:
                data.append(bag)
                bag = Bag()
            last_bag = cur_bag
        # rel label
        bag.rel.append(_rel)

        # type
        _type = [type2id[ins['sub']['type']], type2id[ins['obj']['type']]]
        bag.type.append(_type)

        # word
        words = ins['text'].split()
        _ids = [word2id[word] if word in word2id else word2id['[UNK]'] for word in words]
        _ids = _ids[:opt['max_length']]
        _ids.extend([word2id['[PAD]'] for _ in range(opt['max_length'] - len(words))])
        bag.word.append(_ids)

        # sentence length
        _length = min(len(words), opt['max_length'])
        bag.length.append(_length)

        # ent
        ent1 = ins['sub']['name']
        ent2 = ins['obj']['name']
        _ent1 = word2id[ent1] if ent1 in word2id else word2id['[UNK]']
        _ent2 = word2id[ent2] if ent2 in word2id else word2id['[UNK]']
        bag.ent1.append(_ent1)
        bag.ent2.append(_ent2)

        # pos
        p1 = min(words.index(ent1) if ent1 in words else 0, opt['max_length'] - 1)
        p2 = min(words.index(ent2) if ent2 in words else 0, opt['max_length'] - 1)
        _pos1 = np.arange(opt['max_length']) - p1 + opt['max_pos_length']
        _pos2 = np.arange(opt['max_length']) - p2 + opt['max_pos_length']
        _pos1[_pos1 > 2 * opt['max_pos_length']] = 2 * opt['max_pos_length']
        _pos2[_pos2 > 2 * opt['max_pos_length']] = 2 * opt['max_pos_length']
        _pos1[_pos1 < 0] = 0
        _pos2[_pos2 < 0] = 0
        _pos1[_length:] = 2 * opt['max_pos_length'] + 1
        _pos2[_length:] = 2 * opt['max_pos_length'] + 1
        bag.pos1.append(_pos1)
        bag.pos2.append(_pos2)

        # mask
        p1, p2 = sorted((p1, p2))
        _mask = np.zeros(opt['max_length'], dtype=np.long)
        _mask[p2 + 1: _length] = 3
        _mask[p1 + 1: p2 + 1] = 2
        _mask[:p1 + 1] = 1
        bag.mask.append(_mask)

    # append the last bag
    if last_bag is not None:
        data.append(bag)

    logger.info("Finish "+split+" pre-processing")
    logger.info("Storing processed files...")
    pickle.dump(data, open(os.path.join('processed_data', split + '.pkl'), 'wb'))
    logger.info("Finish "+split+" storing")

if __name__ == '__main__':
    if not os.path.exists("processed_data"):
        os.mkdir("processed_data")

    opt = vars(config())
    logger = get_logger(opt['log_dir'], 'preprocess_result.log')

    splits = ["train","test"]
    for split in splits:
        preprocess(split,opt)
    print("Congratulations! Finish train and test data pre-processing!")
