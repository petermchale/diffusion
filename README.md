## Acknowledgement

This is a lightly edited version of: 

https://www.youtube.com/watch?v=S_il77Ttrmg&list=WL&index=4


## Diffusion model 

A UNet is used to model the mapping from noisy image to denoised image (see `model.py`).

## Sanity-check diffusion model 

I overfitted the diffusion model to a single training image (`racoon.jpg`), and checked that the model generates exactly this image, 
when presented with random noise (`racoon.ipynb`).

## Generating CIFAR10-like images

I then trained the model on CIFAR10 using a single-GPU `g4dn.xlarge` EC2 instance for 160 epochs over a period of about 6 hours. 
Some statistics from the training run can be seen at: https://api.wandb.ai/links/peter-thomas-mchale/76bqm8a8
The model is about 240M in size. 
Finally, I used the trained model to generate new images,
some of which look realistic (`generate-from-cifar10.ipynb`).





