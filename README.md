# BOBBY2: Buffer Based Robust High-Speed Object Tracking
Original implementation of BOBBY2, a sparse Siamese single object tracker with time-delayed feature buffer.    

![BOBBY2 architecture](https://github.com/datacrisis/BOBBY2/blob/master/doc_imgs/BOBBY_2_architecture.png)

Preprint available: [https://arxiv.org/abs/1910.08263]

- - - -
### Introduction
This is an original implementation of BOBBY2 with part of the research source-code.

<p align="justify">
<b>Abstract</b> In this work, a novel high-speed single object tracker that is robust against non-semantic distractor exemplars is introduced; dubbed BOBBY2. It incorporates a novel exemplar buffer module that sparsely caches the target's appearance across time, enabling it to adapt to potential target deformation. As for training, an augmented ImageNet-VID dataset was used in conjunction with the one cycle policy, enabling it to reach convergence with less than 2 epoch worth of data. For validation, the model was benchmarked on the GOT-10k dataset and on an additional small, albeit challenging custom UAV dataset collected with the TU-3 UAV. We demonstrate that the exemplar buffer is capable of providing redundancies in case of unintended target drifts, a desirable trait in any middle to long term tracking. Even when the buffer is predominantly filled with distractors instead of valid exemplars, BOBBY2 is capable of maintaining a near-optimal level of accuracy. BOBBY2 manages to achieve a very competitive result on the GOT-10k dataset and to a lesser degree on the challenging custom TU-3 dataset, without fine-tuning, demonstrating its generalizability. In terms of speed, BOBBY2 utilizes a stripped down AlexNet as feature extractor with 63% less parameters than a vanilla AlexNet, thus being able to run at a competitive 85 FPS.
 </p>


### Setup and Pre-Requisites
- python == 3.7
- pytorch == 1.2
- torchvision
- numpy
- pillow
- lr finder (for training only; credits to David Silva https://github.com/davidtvs/pytorch-lr-finder)
- onecycle lr (for training only; credits to Nachiket Tanksale https://github.com/nachiket273/One_Cycle_Policy)
- ImageNet-VID dataset (for training only; credits to Russakovsky et al. http://image-net.org/challenges/LSVRC/2017/#vid)
- GOT-10k toolkit (for tracking only; credits to Lianghua Huang https://github.com/got-10k/toolkit)
- GOT-10k dataset (for tracking only; credits to Huang et al. http://got-10k.aitestunion.com/downloads)


### Pretrained Model

Download the model from our paper: https://drive.google.com/open?id=1xaGH5k7SMB_gbZKz1vQsR9lq3aX7xt_q

- - - -
### Getting started

**Tracking on GOT-10k**  <br />
*Perform tracking with our pre-trained model on the GOT-10k without training.*
  1. Check and fulfill pre-requisites.
  2. Clone the repository.
  3. Download the pretrained model from the link above and extract it to the `weights` folder.
  4. Download and extract the GOT-10k dataset to a preferred location, dubbed `sequence` herein.
  5. Run `demo_got10k.py` with the proper arguments. Simplified base argument to run demo:
     ```
     python demo_got10k.py -dt sequence -id some_id -m weights -v True
     ```

 **Training your own model** <br />
 *Start here to train your own network.*
  1. Check and fulfill pre-requisites.
  2. Clone the repository.
  3. Download and extract the Imagenet-VID dataset to a preferred location.
  4. Modify paths in `train.py` to correspond to your setup.
  5. Use `lr_finder` to check for optimal learning rate for training.
  6. Train.
  7. Continue (4) onwards in the above tracking instruction to test your tracker.


### Notes on ImageNet-VID Tweak for Training

Due to the unique buffer module, custom generation of positive-negative samples and the way the sample fetching mechanism was implemented, the csv files in `ilsvrc_csv` will be needed for training. If you would like to train as we did, modify the default ImageNet-VID dataset exactly in accordance to the csv files.`ilsvrc_csv` contains 2 directory, `ilsvrc_train_csv` and `ilsvrc_val_csv`. Each of the directory in turn contains 4 csv files used by the custom dataset class in `data_utils` for training. The csv files are used to dynamically generate negative samples and fetches the proper negative-positive sample as needed by the training sequence. Check the docstring in `data_utils` for more info on the individual csv. **Note that you will have to parse and modify the individual path entries in the csv to match your own setup for it to work.**

### Demo on Webcam

First Frame (Initialization, t=0)           |  Subsequent Frames (Tracking, t>0)
:-------------------------:|:-------------------------:
![Cam Demo Gif](https://github.com/datacrisis/BOBBY2/blob/master/doc_imgs/cam_demo_first_frame.png) | ![Cam Demo Gif](https://github.com/datacrisis/BOBBY2/blob/master/doc_imgs/cam_demo_sample.gif)

*Perform tracking with our pre-trained model without training through your webcam.*
  1. Check and fulfill pre-requisites.
  2. Clone the repository.
  3. Download the pretrained model from the link above and extract it to the `weights` folder.
  4. Run `show_cam.py` and the first window (initliazation) will pop up. 
  5. Place your face in the box with as tight as possible and then press `Esc` to close the initialization window.
  6. The second window (tracking) will pop up. Press `Esc` again to terminate when you are done.

- - - -
### License
Licensed under MIT license.


#### Written by
Keifer Lee @ 2019
