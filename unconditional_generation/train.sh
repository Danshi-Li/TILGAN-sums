if [ ! -d "./results" ]; then
  mkdir ./results
fi

CUDA_VISIBLE_DEVICES=0 python train.py --data_path ./data/MS_COCO_right --no_earlystopping --maxlen 15 \
--save ./results/ms_coco_right_klgan --gan_type kl --batch_size 128 --z_size 100 --niters_ae 1 \
--niters_gan_ae 1 --niters_gan_d 1 --niters_gan_g 1 --lr_ae 0.6 --lr_gan_e 1e-4 --lr_gan_g 1e-4 \
--lr_gan_d 1e-4 --add_noise --gan_d_local --enhance_dec


CUDA_VISIBLE_DEVICES=0 python train.py --data_path ./data/NewsData --no_earlystopping --maxlen 32 \
--save ./results/newsdata_klgan --gan_type kl --batch_size 128 --z_size 100 --niters_ae 1 \
--niters_gan_ae 1 --niters_gan_d 1 --niters_gan_g 1 --lr_ae 0.432 --lr_gan_e 1e-4 --lr_gan_g 1e-4 \
--lr_gan_d 1e-4 --add_noise --gan_d_local --enhance_dec