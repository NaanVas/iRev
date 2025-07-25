# for dataset in Tucso_data  Office_Products_data Musical_Instruments_data
for dataset in Musical_Instruments_data
do

    #python3 main.py evaluate --model=DeepCoNN --dataset=$dataset --emb_opt=word2vec --num-fea=1 --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=NARRE --dataset=$dataset --emb_opt=word2vec --num-fea=2 --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=MPCN --dataset=$dataset --emb_opt=word2vec --num-fea=1 --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=D_ATTN --dataset=$dataset --emb_opt=word2vec --num-fea=1 --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=DAML --dataset=$dataset --emb_opt=word2vec --num-fea=2 --batch_size=8 --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=ConvMF --dataset=$dataset --emb_opt=word2vec --num-fea=1 --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=TRANSNET --dataset=$dataset --emb_opt=word2vec --num-fea=1 --output=fm --transnet=True --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=ANR --dataset=$dataset --emb_opt=word2vec --num-fea=1 --id_emb_size=500 --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=HRDR --dataset=$dataset --emb_opt=word2vec --num-fea=2 --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=TARMF --dataset=$dataset --emb_opt=word2vec --num-fea=2 --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=CARL --dataset=$dataset --num-fea=3 --emb_opt=word2vec --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=ALFM --dataset=$dataset --num_fea=1 --topics=True --direct_output=True --emb_opt=word2vec --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=A3NCF --dataset=$dataset --num_fea=1 --topics=True --direct_output=True --emb_opt=word2vec --statistical_test=True --ranking_metrics=True

    python3 main.py evaluate --model=CARP --dataset=$dataset --output=lfm --emb_opt=word2vec --statistical_test=True  --ranking_metrics=True

    #python3 main.py evaluate --model=CARM --dataset=$dataset --emb_opt=word2vec --statistical_test=True --ranking_metrics=True

    #python3 main.py evaluate --model=MAN --dataset=$dataset --batch_size=64 --man=True --emb_opt=word2vec --output=nfm
done