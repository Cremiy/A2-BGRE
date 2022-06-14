# A2-BGRE
The source code for paper DBGARE: Accross-Within Dual Bipartite Graph Attention for Enhancing Distantly Supervised Relation Extraction(KSEM2022)

## Requirements

* Python 3.7
* Pytorch 1.9.1
* PyTorch Geometric 1.7.2

## Dataset
the dataset for our paper can be download from [here](https://drive.google.com/drive/folders/1adFDy20HomzTTfDuGwmHi4omt5pFmCYl?usp=sharing), and put it to `data` folder.
If you would like to use preprocessed data, please check [here](https://drive.google.com), and put it to `processed_data` folder.

### raw data includes:
* train.json
* test.json
* type2id.json
* rel2id.json
* constraint_graph.json(a bipartite graph)
* vec.txt
### processed data includes:
* train.pkl
* test.pkl
* word2id.json
* word_vec.npy
* long_tail.json(for evaluation of long-tailed distributions)

## Train
### If you have not preprocessed the raw data or obtained the preprocessed data failed, please run: 
```
python preprocess_raw_data.py
```
then you will get the preprocessed data in folder `processed_data` easily.

### Train the models:
```
python train.py --model_type main_model --name <model name> --lr 0.5 --batch_size 256 --dropout 0.5
e.g.: python train.py --model_type main_model --name A2BGRE --lr 0.5 --batch_size 256 --dropout 0.5
```
you can adjust the parameters by yourself. default: model_type=main_model lr=0.5,batch_size=256,dropout 0.5, name=A2BGRE

## Test
### Test the models:
Download the pretrained model from [here](https://drive.google.com/drive/folders/13YAj30BGK14oejVsM_9aFtzfT-m8nzUW?usp=sharing), 
and put it to `pretrained_models` folder.
```
python eval.py --model_type main_model --name <model name>
e.g.: python eval.py --model_type main_model --name A2BGRE
```
## Results
* Experimental logger info is in `results/xxx.log`
* Pretrained models is in `pretrained_models`
* Precision/Recall.npy and pr-curve figure are in `results/`

## Citation
```
When our paper is open, we appreciate that you can cite our paper in this citation or citation in google scholar.
```






