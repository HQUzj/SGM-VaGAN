# Spiking Generative Models Based on Variational Autoencoder and Adversarial Training

## Requirements

To install requirements:

```setup
conda env create -f SGM-VAGAN_env.yml
conda activate Test
```

## File

    datasets/  # container of data  
    snn_model/ # snn layers   
    train.py   # training code
    vagan.py   # Spiking Model and Discriminator
    utils/     # files used to evaluate the model
    vizs/      # runing result  

## Training

To train the model(s) in the paper, run this command:  

    __params:__
    -nepch    # training epoch, default is 50
    -warmup   # warmup-epocch use traditional VAE
    -datasets # data set name, default is mnist
    -glr      # learning rate of Spiking model default is 1e-3
    -dlr      # learning rate of discriminator default is 1e-3

```train
python train.py -datasets MNIST --nepch 50 -warmup 1
```



