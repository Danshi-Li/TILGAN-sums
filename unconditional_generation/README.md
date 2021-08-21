# Unconditional Generation Code

This is the implementaition for Unconditional Generation Task.
The model is based on a previous paper entitled  `TILGAN: Transformer-based Implicit Latent GAN for Diverse and Coherent Text Generation`

## Autoencoder backbone experiments

This work explores multiple settings to implement the encoder-decoder backbone model:
| Backbone architecture | Train script file | Backbone class |
|-|-|-|
| Transformer + Transformer | train.py | models.py -> Seq2Seq |
| BERT + Transformer | train_enc_bert.py | models.py -> AE_enc_BERT |
| Transformer + GPT | train_dec_gpt.py | models.py -> AE_dec_GPT |
| BERT + GPT | train_full.py | models.py -> AE_BG |
| T5 | train_t5.py | T5model.py -> AE_T5 |

## Run train script
Take the example as to train with T5 backbone autoencoder:

```
>>> python train_T5.py --data_path data/MS_COCO_right --maxlen 16 --save PretrainGenerator100.0.1\
--batch_size 64 --emsize 512 --nlayers 2 --nheads 4 --aehidden 56 \
--niters_gan_d 1 --niters_gan_ae 1 --lr_gan_g 4e-04 --lr_ae 0.1 --add_noise --gan_d_local
```

## Pretrain generator module
When using the pre-trained autoencoder backbones, the randomly initialized generator needs to be pretrained in order to match the autoencoder for capability.
Use argument ```pretrain_generator [niter]``` to pretrain the generator for ```[niter]``` steps.

## TODOs
Debug the implementation of T5 code  
Study and implement the BART version of encoder-decoder backbone  
Incorporate the idea of the paper ```Prefix Tuning```
