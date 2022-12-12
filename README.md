# Deep Learning for Wireless Networked Systems: a joint Estimation-Control-Scheduling Approach
PyTorch implementation of the deep learning (DL)-based estimator-control-scheduler co-design for a model-unknown nonlinear WNCS over wireless fading channels.
If you use our code or data please cite the paper: https://arxiv.org/pdf/2210.00673.pdf.
### Authors: 
Zihuai Zhao, Wanchun Liu, *Member, IEEE,* Daniel E. Quevedo, *Fellow, IEEE,* Yonghui Li, *Fellow, IEEE,* Branka Vucetic, *Fellow, IEEE* 
### Abstract: 
Wireless networked control system (WNCS) connecting sensors, controllers, and actuators via wireless communications is a key enabling technology for highly scalable and low-cost deployment of control systems in the Industry 4.0 era.
Despite the tight interaction of control and communications in WNCSs, most existing works adopt separative design approaches.
This is mainly because the co-design of control-communication policies requires large and hybrid state and action spaces, making the optimal problem mathematically intractable and difficult to be solved effectively by classic algorithms.
In this paper, we systematically investigate deep learning (DL)-based estimator-control-scheduler co-design for a model-unknown nonlinear WNCS over wireless fading channels.
In particular, we propose a co-design framework with the awareness of the sensor's age-of-information (AoI) states and dynamic channel states.
We propose a novel deep reinforcement learning (DRL)-based algorithm for controller and scheduler optimization utilizing both model-free and model-based data. An AoI-based importance sampling algorithm that takes into account the data accuracy is proposed for enhancing learning efficiency.
We also develop novel schemes for enhancing the stability of joint training. 
Extensive experiments demonstrate that the proposed joint training algorithm can effectively solve the estimation-control-scheduling co-design problem in various scenarios and provide significant performance gain compared to separative design and some benchmark policies.
## Usage
Learning curves and numerical results can be found in the paper. The paper results can be reproduced with minor differences by running:
```
main.ipynb
```
Please note that `main.ipynb` is no longer exactly representative of the code used in the paper. Hyper-parameters can be modified with different arguments to `main.ipynb`. 
## BibTex
```
@article{zhao2022deep,
  title={Deep Learning for Wireless Networked Systems: a joint Estimation-Control-Scheduling Approach},
  author={Zhao, Zihuai and Liu, Wanchun and Quevedo, Daniel E and Li, Yonghui and Vucetic, Branka},
  journal={arXiv preprint arXiv:2210.00673},
  year={2022}
}
```
