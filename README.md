# Selected residual attentive patterns for lightweight networks in image classification
**Abstract:**
Modern deep networks often rely on attention modules which are still at a modest level
due to using either one type of channel-wise patterns or an expensive combination of two
types of them. In case of using all of those, the resultant weight volume has been less
discriminative information due to two separate excitations along with a double increase of
complexity. To deal with these limitations, an efficient attention module is proposed by
addressing two novel components of residual attentive information for a given tensor as
follows: i) top-n channel-residual attentive patterns with a unitary excitation perceptron,
and ii) multiple spatial-residual attentive features. A simple fusion of these complementary
components forms a robust volume of selected residual attentive patterns (named SRAP).
To the best of our knowledge, it is the first time that a channel/spatial-residual mechanism
is proposed to reinforce the attentive information. Experiments on benchmark datasets
for image classification have proved the prominent performance of SRAP versus other
attention modules. Particularly, SRAP improved the performance of concerning networks
by up to ∼7% on ImageNet-100 without increasing the computational complexity.

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
  journal      = {Pattern Recognition Letters},
  note         = {Submitted 2025}
}
```
