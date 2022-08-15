# Multimodal Across Domains Gaze Target Detection
Official PyTorch implementation of "Multimodal Across Domains Gaze Target Detection" at [ICMI 2022](https://icmi.acm.org/2022/).
![An image of our neural network](/assets/network.png?raw=true)

## Requirements
### Environment
To run this repo create a new conda environment and configure all environmental variables using the provided templates.

```bash
conda env create -f environment.yml

cp .env.example .env
nano .env
```

Due to the complexity of the network use a recent NVidia GPU with at least 6GB of memory available and CUDA 11.3+ installed.
Also, we suggest running everything on a Linux-based OS, preferably Ubuntu 20.04.

### Datasets
This network was trained and evaluated on three popular datasets: GazeFollow (extended), VideoAttentionTarget, and GOO (real).
We further extended each sample with depth data. You can download the preprocessed datasets with depth [here]().

## Train and evaluate
Before training, download the pretraining weights [here]().
The script allows to train and evaluate different datasets.
To train and evaluate on the same dataset sets the ‵source_dataset‵ and ‵target_dataset‵ to the same value.
To evaluate only, set the ‵eval_weights‵ variable.

```bash
python main.py [-h] [--tag TAG] [--device {cpu,cuda,mps}] [--input_size INPUT_SIZE] [--output_size OUTPUT_SIZE] [--batch_size BATCH_SIZE]
               [--source_dataset_dir SOURCE_DATASET_DIR] [--source_dataset {gazefollow,videoattentiontarget,goo}] [--target_dataset_dir TARGET_DATASET_DIR]
               [--target_dataset {gazefollow,videoattentiontarget,goo}] [--num_workers NUM_WORKERS] [--init_weights INIT_WEIGHTS] [--eval_weights EVAL_WEIGHTS] [--lr LR]
               [--epochs EPOCHS] [--evaluate_every EVALUATE_EVERY] [--save_every SAVE_EVERY] [--print_every PRINT_EVERY] [--no_resume] [--output_dir OUTPUT_DIR] [--amp AMP]
               [--freeze_scene] [--freeze_face] [--freeze_depth] [--head_da] [--rgb_depth_da] [--task_loss_amp_factor TASK_LOSS_AMP_FACTOR]
               [--rgb_depth_source_loss_amp_factor RGB_DEPTH_SOURCE_LOSS_AMP_FACTOR] [--rgb_depth_target_loss_amp_factor RGB_DEPTH_TARGET_LOSS_AMP_FACTOR]
               [--adv_loss_amp_factor ADV_LOSS_AMP_FACTOR] [--no_wandb] [--no_save]

optional arguments:
  -h, --help            show this help message and exit
  --tag TAG             Description of this run
  --device {cpu,cuda,mps}
  --input_size INPUT_SIZE
                        input size
  --output_size OUTPUT_SIZE
                        output size
  --batch_size BATCH_SIZE
                        batch size
  --source_dataset_dir SOURCE_DATASET_DIR
                        directory where the source dataset is located
  --source_dataset {gazefollow,videoattentiontarget,goo}
  --target_dataset_dir TARGET_DATASET_DIR
                        directory where the target dataset is located
  --target_dataset {gazefollow,videoattentiontarget,goo}
  --num_workers NUM_WORKERS
  --init_weights INIT_WEIGHTS
                        initial weights
  --eval_weights EVAL_WEIGHTS
                        If set, performs evaluation only
  --lr LR               learning rate
  --epochs EPOCHS       number of epochs
  --evaluate_every EVALUATE_EVERY
                        evaluate every N epochs
  --save_every SAVE_EVERY
                        save model every N epochs
  --print_every PRINT_EVERY
                        print training stats every N batches
  --no_resume           Resume from a stopped run if exists
  --output_dir OUTPUT_DIR
                        Path to output folder
  --amp AMP             AMP optimization level
  --freeze_scene        Freeze the scene backbone
  --freeze_face         Freeze the head backbone
  --freeze_depth        Freeze the depth backbone
  --head_da             Do DA on head backbone
  --rgb_depth_da        Do DA on rgb/depth backbone
  --task_loss_amp_factor TASK_LOSS_AMP_FACTOR
  --rgb_depth_source_loss_amp_factor RGB_DEPTH_SOURCE_LOSS_AMP_FACTOR
  --rgb_depth_target_loss_amp_factor RGB_DEPTH_TARGET_LOSS_AMP_FACTOR
  --adv_loss_amp_factor ADV_LOSS_AMP_FACTOR
  --no_wandb            Disables wandb
  --no_save             Do not save checkpoint every {save_every}. Stores last checkpoint only to allow resuming
```
