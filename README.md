# ECE685_final

The base dataset can be downloaded from: https://drive.google.com/file/d/1NL2s-7iTs8kLz0JZKVN3IA1IuQHqmBMS/view?usp=drive_link\
The target task 1 - bone-fracture can be downloaded from: https://drive.google.com/file/d/1wG91SVFca8m1RgnAOvsyjfVZEULPIWyM/view?usp=drive_link\
The target task 2 - shoulder-elbow can be downloaded from: https://drive.google.com/file/d/1NL2s-7iTs8kLz0JZKVN3IA1IuQHqmBMS/view?usp=drive_link\

The following links are small samples for fine tuning: \
bone_fracture_sample: https://drive.google.com/file/d/1m5AfT6jOYRTv-JplRsTgqw5UHlBE3YAf/view?usp=sharing \
:shoulder_elbow_sample https://drive.google.com/file/d/1AlJd8aey9HAHinDOUAUr79pai5592ZXL/view?usp=sharing

### Results Visualization:
All the results are stored in /runs/detect folder. Folder names indicate which experiment we implemented.

### Implementation of MAML:
The implementation of MAML is based on the source code provided by ultralytics. The trainer.py file in /ultralytics/engine is modified to satisfy MAML training process. Optimizer, dataloader, schedular, and training on query set are added in _do_train function. To re-use the code, replace the original source code with the trainer.py file in this repo. In the file, search keyword "project" can locate modifications.