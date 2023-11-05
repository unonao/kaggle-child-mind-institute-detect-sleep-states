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
```