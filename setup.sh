cd ..
git clone --depth 1 https://github.com/nerfstudio-project/gsplat
git clone --depth 1 https://github.com/rahul-goel/fused-ssim

pip install ninja

pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121

pip install pycolmap viser imageio[ffmpeg] scikit-learn tqdm \
  torchmetrics[image] opencv-python Pillow tensorboard tensorly \
  pyyaml matplotlib kornia easydict plotly plyfile tyro

pip install nerfview==0.0.2
pip install numpy==1.24.4
pip install evo==1.11.0
