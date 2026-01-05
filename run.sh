CUDA_VISIBLE_DEVICES=0 \
  python src/trainer.py \
  mcmc \
  --data_dir ${TNT_ROOT}/Truck \
  --data_factor 2 \
  --result_dir \
  results/mcmc-mlp-epi/Truck \
  --pose_opt_type mlp \
  --use_corres_epipolar_loss
