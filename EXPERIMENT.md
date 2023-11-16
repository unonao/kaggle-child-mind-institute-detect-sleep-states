# 実験Commandのメモ

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

# exp007ベース
python run/train.py exp_name=exp009_0 split=stratify_fold_0 batch_size=32 feature_extractor=CNNSpectrogram "pos_weight=[1.0, 5.0, 5.0]" "label_weight=[0.1, 1.0, 1.0]"
python run/train.py exp_name=exp010_0 split=stratify_fold_0 batch_size=32 feature_extractor=CNNSpectrogram "pos_weight=[1.0, 5.0, 5.0]" "label_weight=[0.0, 1.0, 1.0]"

python run/cv_train.py exp_name=exp011 "pos_weight=[1.0, 5.0, 5.0]" 
python run/cv_score.py exp=exp011 post_process.remove_periodicity=false # 0.7368
python run/cv_score.py exp=exp011 post_process.remove_periodicity=true # 0.7493　→ score: 0.7503  →(前処理改良版)　0.7503


python run/cv_train.py exp_name=exp012 "pos_weight=[1.0, 5.0, 5.0]" "features=001"
python run/cv_score.py exp=exp012 post_process.remove_periodicity=false # 0.7317
python run/cv_score.py exp=exp012 post_process.remove_periodicity=true # 0.7442 → score: 0.7444


python run/cv_train.py exp_name=exp013 "pos_weight=[1.0, 5.0, 5.0]" "features=002" # 0.7393
python run/cv_score.py exp=exp013 post_process.remove_periodicity=false  # score: 0.7401
python run/cv_score.py exp=exp013 post_process.remove_periodicity=true #  score: 0.7561 →(前処理改良版)　0.7566
python run/cv_score_one.py +fold=0 exp_name=exp013 post_process.remove_periodicity=true

python run/cv_train.py exp_name=exp014 "pos_weight=[1.0, 5.0, 5.0]" "features=003" # 0.7357
python run/cv_score.py exp=exp014 post_process.remove_periodicity=true # score: 0.7493

# from 013, anglez, enmo の正規化をシリーズごとに
python run/cv_train.py exp_name=exp015 "pos_weight=[1.0, 5.0, 5.0]" "features=004"

# from 013
python run/cv_train.py exp_name=exp016 "pos_weight=[1.0, 5.0, 5.0]" "features=002" "ignore=001"

# periodicityを入れてよしなに学習
python run/cv_train.py exp_name=exp017 "pos_weight=[1.0, 5.0, 5.0]" "features=005" # 0.75377
python run/cv_score.py exp=exp017 post_process.remove_periodicity=false # score: 0.7545
python run/cv_score.py exp=exp017 post_process.remove_periodicity=true # score: 0.7560

python run/cv_train.py exp_name=exp018 "pos_weight=[1.0, 5.0, 5.0]" "features=002" datamodule.how=stride batch_size=50 epoch=20


python run/cv_train.py exp_name=exp019 "pos_weight=[1.0, 5.0, 5.0]" "features=002"  batch_size=50


# 損失計算
python run/cv_train.py exp_name=exp020 "pos_weight=[1.0, 5.0, 5.0]" "features=002" loss=tolerance
python run/cv_train.py exp_name=exp022 "pos_weight=[1.0, 5.0, 5.0]" "features=002" loss=tolerance_nonzero "loss.loss_weight=[1.0, 0.5]"
python run/cv_train.py exp_name=exp021 "pos_weight=[1.0, 5.0, 5.0]" "features=002" loss=tolerance "loss.loss_weight=[1.0, 0.5]"


python run/cv_train.py exp_name=exp023 "features=002" loss=focal  optimizer.lr=0.005

python run/cv_train.py exp_name=exp024 "features=002" loss=focal_bce "loss.weight=[1.0, 1.0]"

python run/cv_train.py exp_name=exp026 "features=002" loss=focal_bce "loss.weight=[0.5, 10.0]"
python run/cv_train.py exp_name=exp027 "features=002" loss=focal_bce "loss.weight=[0.0, 10.0]"
python run/cv_train.py exp_name=exp028 "features=002" loss=focal_bce "loss.weight=[0.0, 10.0]" loss.gamma=1.5



# python run/cv_train.py exp_name=exp029 "pos_weight=[1.0, 5.0, 5.0]" "features=006" # 0.7393　→　?
python run/cv_train.py exp_name=exp029 "pos_weight=[1.0, 5.0, 5.0]" "features=007" # 0.7393　→　?
python run/cv_score.py exp_name=exp029 post_process.remove_periodicity=true # 0.7599
python -m run.cv_inference exp_name=exp029 model.encoder_weights=null phase=train batch_size=8 "features=007" num_tta=2 # 2:0.7615→0.7728, 5:0.7668→0.7765


python run/cv_train.py exp_name=exp020 "pos_weight=[1.0, 5.0, 5.0]" "features=002" loss=tolerance
```
