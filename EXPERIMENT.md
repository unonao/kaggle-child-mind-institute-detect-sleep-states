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


python run/cv_train.py exp_name=exp046 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=007" model=Spec2DCNNSplit model.n_split=4 epoch=30 monitor=val_score monitor_mode=max duration=17280
python run/cv_train.py exp_name=exp045 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=007" model=Spec2DCNN2DayV2 epoch=30 monitor=val_score monitor_mode=max duration=17280


python run/cv_train.py exp_name=exp047 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=007" model=Spec2DCNNSplit model.n_split=2 epoch=30 monitor=val_score monitor_mode=max duration=17280
python run/cv_train.py exp_name=exp048 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=007" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280


python run/cv_train.py exp_name=exp049_affine "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=007" model=Spec2DCNNAffine epoch=30 monitor=val_score monitor_mode=max duration=17280

python run/cv_train.py exp_name=exp050_minmax "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=007" model=Spec2DCNNMinMax epoch=30 monitor=val_score monitor_mode=max duration=17280

python run/cv_train.py exp_name=exp051_weightavg "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=007" model=SpecWeightAvg epoch=30 monitor=val_score monitor_mode=max duration=17280


python run/cv_train.py exp_name=exp052 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=010" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280
python run/cv_train.py exp_name=exp053 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=011" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280

python run/cv_train.py exp_name=exp054 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280
python run/cv_train.py exp_name=exp055 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=013" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280

python run/cv_train.py exp_name=exp056 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 model.encoder_name=mit_b0

python run/cv_train.py exp_name=exp057 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 model.encoder_name=resnet18
python run/cv_train.py exp_name=exp057_resnet50 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 model.encoder_name=resnet50

python run/cv_train.py exp_name=exp058 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 model.encoder_name=timm-mobilenetv3_small_075

python run/cv_train.py exp_name=exp057_mobliev3_s100 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 model.encoder_name=timm-mobilenetv3_small_100


python run/cv_train.py exp_name=054_overlap "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.overlap=360 datamodule.how=overlap

python run/cv_train.py exp_name=exp059_triplet "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" epoch=30 datamodule.overlap=2160 datamodule.how=overlap monitor=val_score monitor_mode=max duration=6480 bg_sampling_rate=0.1


python run/cv_train.py exp_name=exp059_triplet_loss "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" epoch=30 datamodule.overlap=2160 datamodule.how=overlap monitor=val_score monitor_mode=max duration=6480 bg_sampling_rate=0.1 model=Spec2DCNNOverlap # 9h

 #  TODO negative sampling をなくす or　増やす
python run/cv_train.py exp_name=exp059_no_negative "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" epoch=30 datamodule.overlap=2160 datamodule.how=overlap monitor=val_score monitor_mode=max duration=6480 bg_sampling_rate=0.0 model=Spec2DCNNOverlap


python run/cv_train.py exp_name=exp054_stride "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.how=stride datamodule.train_stride=12000

python run/cv_train.py exp_name=exp054_zero_periodicity "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True

python run/cv_train.py exp_name=exp061_split_large_kernel "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True "feature_extractor.kernel_sizes=[128, 32, 16, 4]"
python run/cv_train.py exp_name=exp062_split_drop "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3
python run/cv_train.py exp_name=exp065_split_drop "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.5


python run/cv_train.py exp_name=exp060_transformer "pos_weight=[1.0, 5.0, 5.0]" batch_size=4 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=6
python run/cv_train.py exp_name=exp063_transformer "pos_weight=[1.0, 5.0, 5.0]" batch_size=4 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=6 decoder.num_layers=6
python run/cv_train.py exp_name=exp064_transformer "pos_weight=[1.0, 5.0, 5.0]" batch_size=4 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=4 decoder.num_layers=2
python run/cv_train.py exp_name=exp066_transformer "pos_weight=[1.0, 5.0, 5.0]" batch_size=4 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=2

# now
python run/cv_train.py exp_name=exp068_transformer "pos_weight=[1.0, 5.0, 5.0]" batch_size=4 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=3 decoder.dropout=0.3 


python run/cv_train.py exp_name=exp069_warmup "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 scheduler.use_warmup=True
python run/cv_train.py exp_name=exp070_warmup "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 scheduler.use_warmup=True bg_sampling_rate=0.3
python run/cv_train.py exp_name=exp071_bg_sampling "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 bg_sampling_rate=0.6
python run/cv_train.py exp_name=exp072_low_sigma "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 sigma=5
python run/cv_train.py exp_name=exp073_sigma8_offset20 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 sigma=8 offset=20
python run/cv_train.py exp_name=exp074_transformer_warmup "pos_weight=[1.0, 5.0, 5.0]" batch_size=4 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=3 decoder.dropout=0.4 scheduler.use_warmup=True optimizer.lr=0.001
python run/cv_train.py exp_name=exp075_transformer_warmup "pos_weight=[1.0, 5.0, 5.0]" batch_size=4 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=3 decoder.dropout=0.4 scheduler.use_warmup=True optimizer.lr=0.005
python run/cv_train.py exp_name=exp076_transformer_warmup "pos_weight=[1.0, 5.0, 5.0]" batch_size=4 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=50 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=3 decoder.dropout=0.2 
python run/train.py -m exp_name=transformer_lr "pos_weight=[1.0, 5.0, 5.0]" batch_size=4 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=50 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=3 decoder.dropout=0.3 "optimizer.lr=0.0001,0.005,0.025"
python run/train.py -m exp_name=transformer_dropout "pos_weight=[1.0, 5.0, 5.0]" batch_size=4 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=50 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=3 "decoder.dropout=0.2,0.4"

python run/cv_train.py exp_name=exp078_lstm "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 feature_extractor=LSTMFeatureExtractor 

python run/train.py -m exp_name=lstm_param2 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 feature_extractor=LSTMFeatureExtractor "feature_extractor.hidden_size=32" "feature_extractor.num_layers=3"

python run/train.py -m exp_name=lstm_param3 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 feature_extractor=LSTMFeatureExtractor "feature_extractor.hidden_size=64" "feature_extractor.num_layers=2"

python run/train.py -m exp_name=lstm_param4 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 feature_extractor=LSTMFeatureExtractor "feature_extractor.hidden_size=64" "feature_extractor.num_layers=3,4"

python run/cv_train.py exp_name=exp079_lstm_64 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 feature_extractor=LSTMFeatureExtractor "feature_extractor.hidden_size=64" "feature_extractor.num_layers=2"

# todo

python run/train.py -m exp_name=lstm_param5 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 feature_extractor=LSTMFeatureExtractor "feature_extractor.hidden_size=128" "feature_extractor.num_layers=2,3,4"


python run/train.py -m exp_name=transformer_ "pos_weight=[1.0, 5.0, 5.0]" batch_size=4 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=50 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=3 optimizer.lr=????? decoder.dropout=?

python run/cv_train.py exp_name=exp073_sigma8 "pos_weight=[1.0, 5.0, 5.0]" batch_size=8 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 sigma=8



python run/cv_train.py exp_name=exp077_transformer "pos_weight=[1.0, 5.0, 5.0]" batch_size=4 "features=012" model=Spec2DCNNSplit model.n_split=1 epoch=50 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=3 decoder.dropout=0.3 
```
