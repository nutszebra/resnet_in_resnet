# What's this
Implementation of Residual Networks In Residual Networks by chainer  

# Dependencies

    git clone https://github.com/nutszebra/resnet_in_resnet.git
    cd resnet_in_resnet
    git submodule init
    git submodule update

# How to run
    python main.py -p ./ -g 0 


# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for some parts.  
* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy.  

* Learning rate  
Initial learning rate is 0.1. Learning rate is divided by 5 at [150, 225] and I totally run 300 epochs.


# Cifar10 result

| network              | total accuracy (%) |
|:---------------------|-------------------:|
| 18-layer + wide RiR  | 94.99              |
| my implementation    | 94.43               |

<img src="https://github.com/nutszebra/resnet_in_resnet/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/resnet_in_resnet/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">


# Reference
Resnet in Resnet: Generalizing Residual Architectures [[1]][Paper]

[paper]: https://arxiv.org/abs/1603.08029 "Paper"
