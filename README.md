# Kaggle Child Mind Institute - Detect Sleep States



## How to use

### Docker

```sh
docker compose biuld
docker compose run --rm kaggle bash # bash に入る
docker compose up # jupyter lab 起動
```


### How to use

#### preprocess
```sh
python run/prepare_dev.py 
python run/prepare_data.py phase=train
python run/prepare_data.py phase=test
```

#### train
```sh
python run/train.py exp_name=exp001 batch_size=32 feature_extractor=CNNSpectrogram
python run/train.py exp_name=exp002 batch_size=32 model=Spec2DCNN feature_extractor=PANNsFeatureExtractor decoder=UNet1DDecoder
python run/train.py exp_name=exp003 batch_size=32 model=Spec2DCNN feature_extractor=LSTMFeatureExtractor decoder=UNet1DDecoder
python run/train.py exp_name=exp004 batch_size=32 model=Spec2DCNN feature_extractor=SpecFeatureExtractor decoder=UNet1DDecoder
python run/train.py exp_name=exp005_0 split=stratify_fold_0 batch_size=32 feature_extractor=CNNSpectrogram
python run/train.py exp_name=exp006_0 split=stratify_fold_0 batch_size=32 feature_extractor=LSTMFeatureExtractor

python run/train.py exp_name=exp005_0 split=stratify_fold_0 batch_size=32 feature_extractor=CNNSpectrogram
python run/train.py exp_name=exp007_0 split=stratify_fold_0 batch_size=32 feature_extractor=CNNSpectrogram "pos_weight=[1.0, 5.0, 5.0]"
python run/train.py exp_name=exp008_0 split=stratify_fold_0 batch_size=32 feature_extractor=CNNSpectrogram "pos_weight=[1.0, 10.0, 10.0]"
```

#### search
```sh
# search th
python run/search.py exp=exp005_0 split=stratify_fold_0 how=threshold # score: 0.6972691106831039, th: 0.0038536733146294715
python run/search.py exp=exp007_0 split=stratify_fold_0 how=threshold # score: 0.7140820749853363, th: 0.0038536733146294715

# search distance
python run/search.py exp=exp005_0 split=stratify_fold_0 how=distance post_process.score_th=0.0038536733146294715 # score: 0.7064557388270514, th: 70
python run/search.py exp=exp006_0 split=stratify_fold_0 how=distance post_process.score_th=0.0038536733146294715 # score: 0.7176154573874453, th: 64
python run/search.py exp=exp007_0 split=stratify_fold_0 how=distance post_process.score_th=0.0038536733146294715 # score: 0.7199185318854219, th: 79
python run/search.py exp=exp008_0 split=stratify_fold_0 how=distance post_process.score_th=0.0038536733146294715 # score: 0.7207154455037633, th: 69

```

#### inference