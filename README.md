# ZeroSpeech 2020

Implementation of our submission to ZeroSpeech 2020 (Hou et al.)

Results: https://zerospeech.com/2020/results.html

The system is composed of a hierarchical VQ-VAE encoder to  discover discrete spoken word units from speech and a MelGAN vocoder to  directly generate speech. They are trained separately. During VQ-VAE  training, we add the speaker id before the VQ-VAE decoder to help reduce the speaker information encoded in the word units. 

### Usage:

1. Run scripts/data_manifest.sh (modify datadir to the root dir of raw dataset)
2. python trainer.py --mode vqvae --language [english/surprise] --ckpt_path [path-to-save-model] --datadir [path-to-the-root-dir-of-dataset]
3. python trainer.py --mode melgan --load_vqvae [path-to-vqvae-ckpts] --language [english/surprise] --ckpt_path [path-to-save-model] --datadir [path-to-the-root-dir-of-dataset]
4. python evaluator.py --language [english/surprise] --datadir [path-to-the-root-dir-of-dataset] --vqvae_model [path-to-vqvae-ckpts] --melgan_model [path-to-melgan-ckpts] --save_path [path-to-save-generated-results]



#### Referred Repositories:

1. vq-vae-2-pytorch: https://github.com/rosinality/vq-vae-2-pytorch  
2. melgan-neurips: https://github.com/descriptinc/melgan-neurips  
3. ZeroSpeech-TTS-without-T: https://github.com/andi611/ZeroSpeech-TTS-without-T  
4. VQ-VAE-Speech: https://github.com/swasun/VQ-VAE-Speech  
5. pytorch-vqvae: https://github.com/houwenxin/pytorch-vqvae  