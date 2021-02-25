# Face parsing
A Pytorch implementation face parsing model trained by CelebAMask-HQ, based on [EHANet](https://github.com/JACKYLUO1991/FaceParsing).
## Dependencies
* Pytorch 1.7.1
* numpy
* Python3
* Pillow
* opencv-python
* tenseorboardX
* pandas
* (Optional) [inplace_abn](https://github.com/mapillary/inplace_abn.git)

## Preprocessing
* Prepare training data: -- download [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ)
* Move the mask folder, the image folder, and `CelebA-HQ-to-CelebA-mapping.txt` under `./Data_preprocessing`
* Run `python g_mask.py` to merge separate labels. Support multiprocess, use `python g_mask.py --num_process 4` for 4 processes.
* Run  `python g_partition.py` to split train set and test set.

## Training
* Run `bash run_train.sh #GPU_USE_INDEX`, for example, `bash run_train.sh 0`. 
* Add `--parallel True` in bash script if you want to use multi-GPU for training.


## Well-trained model
* The model can be downloaded [here](https://drive.google.com/file/d/1neFVTZCWZcCeIoYA7V3i1Kk3DqaK4iei/view?usp=sharing) (96M).
* The model (`#num.pth`) should be put under `./models/FaceParseNet50/`
* Mask labels are defined as following:

|-| Label list (19 classes) |-|
| ------------ | ------------- | ------------ |
| 0: 'background' | 1: 'skin' | 2: 'nose' |
| 3: 'eye_g' | 4: 'l_eye' | 5: 'r_eye' |
| 6: 'l_brow' | 7: 'r_brow' | 8: 'l_ear' |
| 9: 'r_ear' | 10: 'mouth' | 11: 'u_lip' |
| 12: 'l_lip' | 13: 'hair' | 14: 'hat' |
| 15: 'ear_r' | 16: 'neck_l' | 17: 'neck' |
| 18: 'cloth' | - | - |

* Overall Per-pixel Acc: 94.26;
* Mean IoU: 76.64;
* Overall F1 Score: 85.33.
*  **Note**: train and evaluate according to CelebA train/test split.

## Testing & Color visualization
* Run `bash run_test.sh #GPU_num #Model_num`, for example, `bash run_test.sh 0 32`, which uses epoch 32 model file (`32_G.pth`). 
* Pred results will be saved in `./test_pred_results`
* Color visualized results will be saved in `./test_color_visualize`
* Another way for color visualization without using GPU: Run `python ./Data_preprocessing/g_color.py`

## Visual Results
* OriginalImg vs. GroundTruth vs. PredResult
<div><div align=center>
  <img src="https://github.com/TracelessLe/FaceParsing.PyTorch/blob/master/samples/sample1.png" width="900" height="300" alt="sample1_img-gt-pred"/>
  <img src="https://github.com/TracelessLe/FaceParsing.PyTorch/blob/master/samples/sample2.png" width="900" height="300" alt="sample2_img-gt-pred"/>
</div>
  
## References
```
@article{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  journal={arXiv preprint arXiv:1907.11922},
  year={2019},
  website={https://github.com/switchablenorms/CelebAMask-HQ}
}
@article{luo2020ehanet,
  title={EHANet: An Effective Hierarchical Aggregation Network for Face Parsing},
  author={Luo, Ling and Xue, Dingyu and Feng, Xinglong},
  journal={Applied Sciences},
  volume={10},
  number={9},
  pages={3135},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute},
  website={https://github.com/JACKYLUO1991/FaceParsing}
}
@code{face-parsing.PyTorch,
  author={zllrunning},
  year={2019},
  website={https://github.com/zllrunning/face-parsing.PyTorch}
}
```
