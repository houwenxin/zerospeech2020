cd ..
# rm -rf ./ckpts_codebook128_mel256_spk_adv_enc_rec_loss/*

nohup python trainer_copy2.py --mode melgan --load_melgan --ckpt_path ./ckpts_codebook128_mel256_spk_adv_enc_rec_loss/ > ./ckpts_codebook128_mel256_spk_adv_enc_rec_loss/train_log 2>&1 &
