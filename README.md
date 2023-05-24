### Data preprocessing
* Face Reenactment: https://github.com/AliaksandrSiarohin/first-order-model
* FaceSwap: https://github.com/Seanseattle/MobileFaceSwap
* Fab-Net: https://github.com/oawiles/FAb-Net
* AU, Landmarks, Head pose: https://github.com/TadasBaltrusaitis/OpenFace
* VGG-Face, ArcFace: https://github.com/serengil/deepface
### Publicly Available Deepfake Detection
* ID_reveal (official): https://github.com/grip-unina/id-reveal
* Mesonet (unofficial): https://github.com/HongguLiu/MesoNet-Pytorch
* EfficientNet (unofficial): https://github.com/lukemelas/EfficientNet-PyTorch
* Xception (unofficial): https://github.com/tstandley/Xception-PyTorch
* F3-Net (unofficial): https://github.com/yyk-wew/F3Net
* Face-X-ray (unofficial) : https://github.com/neverUseThisName/Face-X-Ray
* FWA (official): https://github.com/yuezunli/CVPRW2019_Face_Artifacts
* PCL+I2G (unofficial): https://github.com/jtchen0528/PCL-I2G
* EFBN4+SBIs (official): https://github.com/mapooon/SelfBlendedImages
* FTCN: https://github.com/yinglinzheng/FTCN
* ICT: https://github.com/LightDXY/ICT_DeepFake
### Datasets
* Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
* VoxCeleb: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
### Usage
1. Download dataset from above links
2. If needed, crop the face using the repo from [ICT](https://github.com/LightDXY/ICT_DeepFake)
3. Generate training dataset from [FOMM](https://github.com/AliaksandrSiarohin/first-order-model)
4. Modify the `cfg.yaml` with your paths
5. Training
```
python train.py cfg.yaml
```
6. Testing
```
python test.py cfg.yaml
```