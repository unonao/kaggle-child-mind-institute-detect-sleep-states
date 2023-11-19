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


python run/cv_train.py exp_name=exp030_tolerance_loss "pos_weight=[1.0, 5.0, 5.0]" "features=007" loss=tolerance

python run/cv_train.py exp_name=exp031_std_diff "pos_weight=[1.0, 5.0, 5.0]" "features=008" #

python run/cv_train.py exp_name=exp032_sigma_decay "pos_weight=[1.0, 5.0, 5.0]" "features=007" sigma_decay=0.95
python run/cv_train.py exp_name=exp033_sleep_decay "pos_weight=[1.0, 5.0, 5.0]" "features=007" sleep_decay=0.90
python run/cv_train.py exp_name=exp034_sleep_decay "pos_weight=[1.0, 20.0, 20.0]" "features=007" sleep_decay=0.95 monitor=val_score monitor_mode=max

python run/cv_train.py exp_name=exp035 "pos_weight=[1.0, 5.0, 5.0]" "features=007" optimizer.lr=0.002

python run/cv_train.py exp_name=exp036 "pos_weight=[1.0, 5.0, 5.0]" "features=007" optimizer.lr=0.001 batch_size=16

python run/cv_train.py exp_name=exp037 "pos_weight=[1.0, 5.0, 5.0]" "features=007" downsample_rate=3
python run/cv_train.py exp_name=exp038_recall "pos_weight=[1.0, 20.0, 20.0]" "features=007"

python run/cv_train.py exp_name=exp039 "pos_weight=[1.0, 5.0, 5.0]" "features=007" downsample_rate=1

python run/cv_train.py exp_name=exp040 "pos_weight=[1.0, 5.0, 5.0]" "features=009"

python run/cv_train.py exp_name=exp041 "pos_weight=[1.0, 5.0, 5.0]" duration=11520 "features=007"


python run/cv_train.py exp_name=exp042 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=007" model=Spec2DCNN2Day epoch=20

python run/cv_train.py exp_name=exp043 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=007" model=Spec2DCNN2DayV2 epoch=15 monitor=val_score monitor_mode=max duration=34560
python run/cv_score.py exp_name=exp043 post_process.remove_periodicity=true post_process.distance=80 # score: 0.7700

python run/cv_train.py exp_name=exp044 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=007" model=Spec2DCNN2DayV2 epoch=15 monitor=val_score monitor_mode=max duration=17280
python run/cv_score.py exp_name=exp044 post_process.remove_periodicity=true post_process.distance=80 #  score: 0.7744
python -m run.cv_inference exp_name=exp044 model.encoder_weights=null phase=train model=Spec2DCNN2DayV2 duration=17280  batch_size=8 "features=007" num_tta=3 # 1:0.7650→0.7744 # 2: 0.7630→0.7724 # 3: 0.7745→0.7814 # 5: 0.7749→0.7811

python run/cv_train.py exp_name=exp045 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=007" model=Spec2DCNN2DayV2 epoch=30 monitor=val_score monitor_mode=max duration=17280

python run/cv_train.py exp_name=exp046 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=007" model=Spec2DCNNSplit model.n_split=4 epoch=30 monitor=val_score monitor_mode=max duration=17280
```
