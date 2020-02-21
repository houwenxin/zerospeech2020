cd ..
# rm -rf ./ckpts_codebook128_mel256_spk_adv_pretrain/*

nohup python trainer_copy2.py --mode melgan_pretrain --load_melgan --ckpt_path ./ckpts_codebook128_mel256_spk_adv_pretrain/ > ./ckpts_codebook128_mel256_spk_adv_pretrain/train_log 2>&1 &
