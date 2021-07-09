# Unconditional Generation Code

This is the implementaition for Unconditional Generation Task.

## Quick Links
- [Unconditional Generation Code](#unconditional-generation-code)
  - [Quick Links](#quick-links)
  - [Environment](#environment)
  - [Quick Start](#quick-start)
  - [Hyper-parameter Setting](#hyper-parameter-setting)
  - [Acknowledgement](#acknowledgement)

## Environment 
To run our code, please install all the dependency packages by using the following command under python 3.6 (recommend):

```
pip install -r requirements.txt
```

## Quick Start
After the environment setup, you can simply type the following command:
  ```shell
  bash train.sh
  ```

Or you can type the commends as following:

   ```shell
   mkdir results
   # Run train.py
   python train.py --data_path ./data/NewsData --no_earlystopping --maxlen 32 --save ./results/newsdata_klgan --gan_type kl --batch_size 256 --z_size 512 --niters_ae 1 --niters_gan_ae 1 --niters_gan_d 5 --niters_gan_g 1 --lr_ae 0.7 --lr_gan_e 1e-4 --lr_gan_g 1e-4 --lr_gan_d 1e-4 
   ```

You can simply add arguments `--add_noise` , `--gan_d_local` and `--enhance_dec` to test the variants of our model.


## Acknowledgement
Thanks to the source code provider [ARAE](https://openreview.net/forum?id=BkM3ibZRW)
