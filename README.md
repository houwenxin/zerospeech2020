# zerospeech2020

Requirements: pytorch, librosa, pysoundfile, tqdm
  
### TODO List:
1. Add matplotlib figure drawing into the code (VQVAE, MelGAN, MelGAN_pretrain).  
2. Add code to upload several audio sample to tensorboard each epoch.  
3. Tune learning rate / batch size / epoch nums to shorten the training time to 1~2 days. 
  
  
### Progress:
1. Pure VQVAE-2 Done. (Plan: only use bottom encoding.)
2. Try Speaker Adversarial Training & Add speaker info to VQVAE decoder. (Done)
3. Train vanilla MelGAN. (Done.)
4. Train MelGAN with weighted loss: a*loss_enc + b*loss_mel + c*loss_adv. (TODO)
5. Pretrain MelGAN with loss_mel. (In progress)
6. Train Pretrained MelGAN with weighted loss: a*loss_enc + b*loss_mel + c*loss_adv. (TODO)

### Related Repositories:
1. vq-vae-2-pytorch: https://github.com/rosinality/vq-vae-2-pytorch  
2. melgan-neurips: https://github.com/descriptinc/melgan-neurips  
3. ZeroSpeech-TTS-without-T: https://github.com/andi611/ZeroSpeech-TTS-without-T  
4. VQ-VAE-Speech: https://github.com/swasun/VQ-VAE-Speech  
5. pytorch-vqvae: https://github.com/houwenxin/pytorch-vqvae  

### References:
+ Unsupervised End-to-End Learning of Discrete Linguistic Units for Voice. https://arxiv.org/pdf/1905.11563.pdf  
+ VQVAE Unsupervised Unit Discovery and Multi-scale Code2Spec Inverter. https://arxiv.org/pdf/1905.11449.pdf  
+ MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis. https://arxiv.org/pdf/1910.06711.pdf  
+ Generating Diverse High-Fidelity Images with VQ-VAE-2. https://arxiv.org/pdf/1906.00446.pdf  
+ [Possible] SPEAKER INVARIANT FEATURE EXTRACTION FOR ZERO-RESOURCE LANGUAGESWITH ADVERSARIAL LEARNING  

