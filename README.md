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
python run/train.py downsample_rate=2 duration=5760 exp_name=exp001 batch_size=32
python run/train.py exp_name=exp002 downsample_rate=2 duration=5760 batch_size=32 model=Spec2DCNN feature_extractor=PANNsFeatureExtractor decoder=UNet1DDecoder
python run/train.py exp_name=exp003 downsample_rate=2 duration=5760 batch_size=32 model=Spec2DCNN feature_extractor=LSTMFeatureExtractor decoder=UNet1DDecoder
python run/train.py exp_name=exp004 downsample_rate=2 duration=5760 batch_size=32 model=Spec2DCNN feature_extractor=SpecFeatureExtractor decoder=UNet1DDecoder
```

#### search
```sh
# search th
python run/search.py exp=exp001 #  score: 0.7498262920021056, th: 0.0038536733146294715
python run/search.py exp=exp002 #  score: 0.7408391055155943, th: 0.0038536733146294715
python run/search.py exp=exp003 #  score: 0.7619025463250211, th: 0.0038536733146294715

# search distance
python run/search.py exp=exp003 train.post_process.score_th=0.005
```