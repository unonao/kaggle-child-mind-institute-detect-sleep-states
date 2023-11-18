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


python run/cv_train.py exp_name=exp013 "pos_weight=[1.0, 5.0, 5.0]" "features=002" # 0.7393
python run/cv_score.py exp=exp013 post_process.remove_periodicity=false  # score: 0.7401
python run/cv_score.py exp=exp013 post_process.remove_periodicity=true #  score: 0.7566
python run/cv_score.py exp=exp013 post_process.remove_periodicity=true post_process.periodicity.filter_size=10000  #  score: 0.7571
python -m run.cv_inference exp_name=exp013 model.encoder_weights=null phase=train batch_size=8 "features=002" num_tta=5  # tta=1:0.7400 → 0.7566 tta=2:0.7572→0.7706 tta=3: 0.7614→0.7725 tta=5: 0.7639→0.7741

python run/cv_train.py exp_name=exp029 "pos_weight=[1.0, 5.0, 5.0]" "features=007" # 0.7393　→　?
python run/cv_score.py exp_name=exp029 post_process.remove_periodicity=true post_process.distance=40 # 80: 0.7599
python -m run.cv_inference exp_name=exp029 model.encoder_weights=null phase=train batch_size=8 "features=007" num_tta=2 # 2:0.7615→0.7728, 5:0.7668→0.7765

``` 

#### inference

```sh
python -m run.cv_inference exp_name=exp013 model.encoder_weights=null phase=test post_process.remove_periodicity=true batch_size=8 "features=002" num_tta=2
```

