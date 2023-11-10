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

# best
python run/cv_train.py epoch=1
python run/cv_train.py exp_name=exp011 "pos_weight=[1.0, 5.0, 5.0]" 
```

#### score
```sh
# score th
python run/score.py exp=exp005_0 split=stratify_fold_0 how=threshold # score: 0.6972691106831039, th: 0.0038536733146294715
python run/score.py exp=exp007_0 split=stratify_fold_0 how=threshold # score: 0.7140820749853363, th: 0.0038536733146294715
python run/score.py exp=exp010_0 split=stratify_fold_0 how=threshold 
# score distance
python run/score.py exp=exp005_0 split=stratify_fold_0 how=distance post_process.score_th=0.0038536733146294715 # score: 0.7064557388270514, distance: 70
python run/score.py exp=exp006_0 split=stratify_fold_0 how=distance post_process.score_th=0.0038536733146294715 # score: 0.7176154573874453, distance: 64
python run/score.py exp=exp007_0 split=stratify_fold_0 how=distance post_process.score_th=0.0038536733146294715 # score: 0.7199185318854219, distance: 79
python run/score.py exp=exp008_0 split=stratify_fold_0 how=distance post_process.score_th=0.0038536733146294715 # score: 0.7207154455037633, distance: 69
python run/score.py exp=exp009_0 split=stratify_fold_0 how=distance post_process.score_th=0.0038536733146294715 # score: 0.7103450691338504, distance: 63
python run/score.py exp=exp010_0 split=stratify_fold_0 how=distance post_process.score_th=0.0038536733146294715 #  score: 0.6744478935205226, distance: 63

python run/score.py exp=exp007_0 split=stratify_fold_0 how=group_by_day 
python run/score.py exp=exp008_0 split=stratify_fold_0 how=group_by_day # score: 0.6601665731898787
```
