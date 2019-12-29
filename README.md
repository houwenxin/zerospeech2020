# zerospeech2020

Requirements: pytorch, librosa, pysoundfile, tqdm
  
### Progress:
1. Pure VQVAE-2 (DONE).
2. Try Speaker Adversarial Training (TODO) & Add speaker info to VQVAE decoder (DONE). (Which is better?)
3. Core idea: Self-Cycle GAN (TODO).

### Developing Log:
2019/12/27: Fix bugs on speaker ids, Sampling rate of Audio2Mel & Mel2Audio. Develop evaluator for generating converted files.  
2019/12/29: Consider a new method: Self-Cyclic Speaker-Independent Voice Conversion (See Training Methods).  

### Training Methods:  
1. Train VQVAE (with Decoder) on all data, then fix VQVAE, train MelGAN on V001 and V002 data (with speaker id).  
2. Train VQVAE (with Decoder) on all data, then do not fix VQVAE, train MelGAN with VQVAE Encoder on V001 and V002 data (with speaker id).  
3. Train VQVAE together with MelGAN on all data (with speaker id).  

### Related Repositories:
1. vq-vae-2-pytorch: https://github.com/rosinality/vq-vae-2-pytorch  
2. melgan-neurips: https://github.com/descriptinc/melgan-neurips  
3. ZeroSpeech-TTS-without-T: https://github.com/andi611/ZeroSpeech-TTS-without-T  
4. VQ-VAE-Speech: https://github.com/swasun/VQ-VAE-Speech  
5. pytorch-vqvae: https://github.com/houwenxin/pytorch-vqvae  
6. cyclevae-vc: https://github.com/patrickltobing/cyclevae-vc  

### References:
+ Unsupervised End-to-End Learning of Discrete Linguistic Units for Voice. https://arxiv.org/pdf/1905.11563.pdf  
+ VQVAE Unsupervised Unit Discovery and Multi-scale Code2Spec Inverter. https://arxiv.org/pdf/1905.11449.pdf  
+ MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis. https://arxiv.org/pdf/1910.06711.pdf  
+ Generating Diverse High-Fidelity Images with VQ-VAE-2. https://arxiv.org/pdf/1906.00446.pdf  
+ Unsupervised speech representation learning using WaveNet autoencoders. https://arxiv.org/pdf/1901.08810.pdf  
+ [Possible] SPEAKER INVARIANT FEATURE EXTRACTION FOR ZERO-RESOURCE LANGUAGESWITH ADVERSARIAL LEARNING. https://tsuchhiii.github.io/pdf/paper/18_icassp_tsuchiya.pdf  
+ [Possible] SPEAKER-INVARIANT TRAINING VIA ADVERSARIAL LEARNING. https://www.microsoft.com/en-us/research/uploads/prod/2018/04/ICASSP2018_Speaker_Invariant_Training.pdf  
+ [Possible] Multi-target Voice Conversion without Parallel Data by Adversarially Learning Disentangled Audio Representations. https://arxiv.org/pdf/1804.02812.pdf  
+ [Possible] Unpaired Image-to-Image Translationusing Cycle-Consistent Adversarial Networks. https://arxiv.org/pdf/1703.10593.pdf  
+ [Possible] Non-Parallel Voice Conversion with Cyclic Variational Autoencoder. https://arxiv.org/pdf/1907.10185v1.pdf  
