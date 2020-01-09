dataset=english_small
cd ../../databases/${dataset}

ls train/unit/*.wav > rec_train_files.txt
ls train/voice/*.wav >> rec_train_files.txt
ls train/voice/*.wav > gan_train_files.txt  
  
ls test/*.wav > eval_files.txt                                                                                                                                                                                      
