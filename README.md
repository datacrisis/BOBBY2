# BOBBY2: Buffer Based Robust High-Speed Object Tracking [Under Construction]
Original implementation of BOBBY2, a sparse Siamese single object tracker with time-delayed feature buffer.    

![BOBBY2 architecture](https://github.com/datacrisis/BOBBY2/blob/master/doc_imgs/BOBBY_2_architecture.png)

Preprint available: [https://arxiv.org/abs/1910.08263]

- - - -
## Introduction
This is an original implementation of BOBBY2 with the research source-code.

**Abstract** In this work, a novel high-speed single object tracker that is robust against non-semantic distractor exemplars is introduced; dubbed BOBBY2. It incorporates a novel exemplar buffer module that sparsely caches the target's appearance across time, enabling it to adapt to potential target deformation. As for training, an augmented ImageNet-VID dataset was used in conjunction with the one cycle policy, enabling it to reach convergence with less than 2 epoch worth of data. For validation, the model was benchmarked on the GOT-10k dataset and on an additional small, albeit challenging custom UAV dataset collected with the TU-3 UAV. We demonstrate that the exemplar buffer is capable of providing redundancies in case of unintended target drifts, a desirable trait in any middle to long term tracking. Even when the buffer is predominantly filled with distractors instead of valid exemplars, BOBBY2 is capable of maintaining a near-optimal level of accuracy. BOBBY2 manages to achieve a very competitive result on the GOT-10k dataset and to a lesser degree on the challenging custom TU-3 dataset, without fine-tuning, demonstrating its generalizability. In terms of speed, BOBBY2 utilizes a stripped down AlexNet as feature extractor with 63% less parameters than a vanilla AlexNet, thus being able to run at a competitive 85 FPS.

## Setup and Pre-Requisites
- python == 3.7
- pytorch == 1.2
- torchvision
- numpy
- pillow
- lr finder (for training only; credits to David Silva https://github.com/davidtvs/pytorch-lr-finder)

## Getting started

[ **Tracking only** ] <br />
*Perform tracking with our pre-trained model without training.*
  1. Contrary to popular belief, Lorem Ipsum is not simply random text
  2. Clone the repository.
  3. Contrary to popular belief, Lorem Ipsum is not simply random text
  4.Contrary to popular belief, Lorem Ipsum is not simply random text
  5. Contrary to popular belief, Lorem Ipsum is not simply random text
  6. Contrary to popular belief, Lorem Ipsum is not simply random text

 [ **Training and tracking** ] <br />
 *Start here to train your own network.*
  1.Contrary to popular belief, Lorem Ipsum is not simply random text
  2. Clone the repository.
  3. Contrary to popular belief, Lorem Ipsum is not simply random text
  4. Contrary to popular belief, Lorem Ipsum is not simply random text
  5. Contrary to popular belief, Lorem Ipsum is not simply random text
  6. Contrary to popular belief, Lorem Ipsum is not simply random text
  7. Contrary to popular belief, Lorem Ipsum is not simply random text (`abc/bac/`).
  8. Contrary to popular belief, Lorem Ipsum is not simply random text

## Pretrained Model

Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.

## ImageNet-VID Tweak Logs

Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.

## Demo and Test on ImageNet-VID
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.

## Demo and Test on GOT-10K
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.

```
cd somehwere
python myheart.py
```

- - - -
## License
Licensed under MIT license.
