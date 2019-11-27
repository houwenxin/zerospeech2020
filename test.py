from trainer import Trainer
from hps.hps import Hps
from dataloader import DataLoader, Dataset

def get_trainer(hps_path, model_path):
	HPS = Hps(hps_path)
	hps = HPS.get_tuple()
	trainer = Trainer(hps, None)
	trainer.load_model(model_path)
	return trainer

if __name__ == "__main__":
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    from hps.hps import hp
    from utils import audio2 as audio
    import soundfile as sf
    import os
    dataset = Dataset(h5py_path="./dataset/english/dataset.hdf5",
            index_path="./dataset/english/index.json",
            load_mel=True,
            load_mfcc=False)
    data_loader = DataLoader(dataset, batch_size=1)
    X = next(data_loader)[2].permute(0, 2, 1)
    trainer = get_trainer(None, "./ckpts/-vqvae-500")
    rec, diff = trainer.vqvae(X)
    rec = rec.permute(0, 2, 1).squeeze()
    print(rec.shape)
    rec = rec.detach().numpy()
    librosa.display.specshow(rec)
    plt.show()
    wav_data_mel = audio.mel2wav(rec, hp)
    wav_path = os.path.join(".", "rebuild_lin.wav")
    sf.write(wav_path, wav_data_mel, hp.sr, 'PCM_16')
    