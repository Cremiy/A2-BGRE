import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='main_model',help='other two choices includes simple_adding,'
                                                                  'simple_concatenating')
    parser.add_argument('--name', default='A2BGRE')
    parser.add_argument('--root', default='data', help='root of data files')
    parser.add_argument('--processed_data_dir', default='processed_data')
    parser.add_argument('--train', default='train.json')
    parser.add_argument('--test', default='test.json')
    parser.add_argument('--rel2id', default='rel2id.json')
    parser.add_argument('--type2id', default='type2id.json')
    parser.add_argument('--graph', default='constraint_graph.json')
    parser.add_argument('--vec_dir', default='vec.txt')
    parser.add_argument('--trained_models_dir', default='pretrained_models')
    parser.add_argument('--log_dir', default='results')
    parser.add_argument('--batch_size', default=256, type=int) #160
    parser.add_argument('--max_length', default=120, type=int)
    parser.add_argument('--max_pos_length', default=100, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--lr', default=0.5, type=float)
    parser.add_argument('--lambda', default=20., type=float)
    parser.add_argument('--val_iter', default=1, type=int)
    parser.add_argument('--pos_dim', default=5, type=int)
    parser.add_argument('--hidden_size', default=230, type=int)
    parser.add_argument('--graph_emb', default=700, type=int)
    parser.add_argument('--graph_dim', default=950, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--seed', default=16051643, type=int)
    return parser.parse_args()