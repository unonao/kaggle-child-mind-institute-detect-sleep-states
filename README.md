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
python run/train.py exp_name=exp007_0 split=stratify_fold_0 batch_size=32 feature_extractor=CNNSpectrogram "pos_weight=[1.0, 5.0, 5.0]"

```

#### inference

```sh
python -m run.inference exp_name=exp007_0 model.encoder_weights=null phase=test post_process.remove_periodicity=true

python -m run.cv_inference exp_name=exp011 model.encoder_weights=null phase=test  post_process.remove_periodicity=true
```
