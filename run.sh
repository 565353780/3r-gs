DATA_FOLDER=$HOME/chLi/Dataset/GS/haizei_1

CUDA_VISIBLE_DEVICES=0 \
  python src/trainer.py \
  mcmc \
  --data_dir ${DATA_FOLDER}/input/ \
  --data_factor 2 \
  --result_dir ${DATA_FOLDER}/input_mcmc-mlp-epi \
  --pose_opt_type mlp \
  --use_corres_epipolar_loss
