# Selected residual attentive patterns for lightweight networks in image classification
**Abstract:**
Attention mechanism allows deep convolutional networks to concentrate on the important patterns instead of on the less useful information during the learning process.
However, the discriminative power of current attention modules is still at a modest level because they have just addressed either one type of global channel-wise patterns or an expensive combination of two kinds of squeezes. In the case of using two types of those, the resultant weight volume has been less discriminative information caused by two separate excitation perceptrons with a double increase of computational complexity.
To deal with these problems, an efficient attention module for lightweight networks is proposed by addressing two novel components of residual attentive information for a given tensor as follows: i) top-n multiple channel-based residual attentive patterns with a unitary excitation perceptron, and ii) multiple spatial-based residual attentive features.
A simple fusion of these two attention components forms a robust volume of selected residual attentive patterns (named SRAP).
To the best of our knowledge, it is the first time that a channel-spatial residual mechanism is proposed to reinforce the attentive information.
Experimental results on benchmark datasets for image recognition have authenticated the prominent performance of SRAP in comparison with other attention modules.

<u>**An example for training SRAP-based models on Stanford Dogs:**</u>


Traning and testing SRAP$^{avg\_std}$ on Stanford Dogs by default. Please change the code in the SRAP.py file for the favorable networks.
The default path of dataset: '../../../datasets/Stanford Dogs'.

For tranining, run command:
```
$ python SRAP_Dogs.py
```
For validating, run command:
```
$ python SRAP_Dogs.py --evaluate
```

Note: For a validation instance of SRAP-based models, a trained model on Stanford Dogs is available at /checkpoints/StanfordDogs/model_best.pth

**Related citation(s):**

If you use any materials, please cite the following relevant works.

```
@article{SRAPAttNguyen25,
  author       = {Thanh Tuan Nguyen, Hoang Anh Pham, Thinh Vinh Le, and Thanh Phuong Nguyen},
  title        = {Selected residual attentive patterns for lightweight networks in image classification},
  journal      = {Computers and Electrical Engineering},
  note         = {Submitted 2025}
}
```
