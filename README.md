## Acknowledgement

This is a lightly edited version of the following implementation of a Denoising Diffusion Probabilistic Model (DDPM): 

https://www.youtube.com/watch?v=S_il77Ttrmg&list=WL&index=4


## Theory 

See `math.ipynb`.

## Diffusion model 

A UNet is used to model the mapping from noisy image to denoised image (see `model.py`).

## Sanity-check diffusion model 

I overfitted the diffusion model to a single training image (`racoon.jpg`), and checked that the model generates exactly this image, 
when presented with random noise (`racoon.ipynb`).

## Generating CIFAR10-like images

I then trained the model on CIFAR10 using a single-GPU `g4dn.xlarge` EC2 instance for 160 epochs over a period of about 6 hours. 
Some statistics from the training run can be seen at: https://api.wandb.ai/links/peter-thomas-mchale/76bqm8a8
The model is about 240Mb in size. 
Finally, I used the trained model to generate new images,
some of which look realistic (`generate-from-cifar10.ipynb`).

## Further improvements 

Most of the images generated from the CIFAR10 model are not realistic. There are two opposite explanations: 

1. The labels were not one-hot encoded (and masked out), as they were in the conditional diffusion model described [here](https://learn.deeplearning.ai/diffusion-models/lesson/6/controlling). It's 
possible that, without masking the label, the model can over-rely upon the labels and not learn how to generate images at all. 
2. The conditional diffusion model we trained above is actually ignoring the labels, 
which could be corrected using classifier or classifier-free guidance, as described in [Luo 2022](https://arxiv.org/abs/2208.11970). 

## Other resources 

See also `https://github.com/petermchale/minDiffusion` 





