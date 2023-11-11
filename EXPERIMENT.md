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
python run/cv_score.py exp=exp011 post_process.remove_periodicity=true # 0.7493　→ score: 0.7503


python run/cv_train.py exp_name=exp012 "pos_weight=[1.0, 5.0, 5.0]" "features=001"
python run/cv_score.py exp=exp012 post_process.remove_periodicity=false # 0.7317
python run/cv_score.py exp=exp012 post_process.remove_periodicity=true # 0.7442 → score: 0.7444


python run/cv_train.py exp_name=exp013 "pos_weight=[1.0, 5.0, 5.0]" "features=002"
python run/cv_train.py exp_name=exp014 "pos_weight=[1.0, 5.0, 5.0]" "features=003"
```
