cd ..
git clone --depth 1 https://github.com/rahul-goel/fused-ssim
git clone --depth 1 https://github.com/rmbrualla/pycolmap

pip install ninja

pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121

pip install pycolmap viser imageio[ffmpeg] scikit-learn tqdm \
  torchmetrics[image] opencv-python Pillow tensorboard tensorly \
  pyyaml matplotlib kornia easydict plotly plyfile tyro gsplat

pip install nerfview==0.0.2
pip install evo==1.11.0
pip install "numpy<2.0"

cd fused-ssim
python setup.py install

cd ../pycolmap
pip install -e .
