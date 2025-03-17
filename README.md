# M3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation
### 📝[Paper](https://arxiv.org/abs/2412.13803) | 🌍[Project Page](https://zixuan-chen.github.io/M-cube-VOS.github.io/) | 🤗[Tools](https://github.com/Lijiaxin0111/SemiAuto-Multi-Level-Annotation-Tool) | 🛢️[Data](https://drive.google.com/drive/folders/1qNSvE6dpkCHSs_8eZRo6vruLScCHl7oI?usp=sharing)

![alt text](./assets/teaser.png)

## 1. Installation
```bash
# Clone this repo
git clone https://github.com/zixuan-chen/M3VOS_Experiment.git
cd M3VOS_Experiment

# Create a Conda environment
conda create -n mvos python=3.10.0
conda activate mvos

# Install pytorch
# Look up https://pytorch.org/get-started/previous-versions/ with your cuda version for a correct command
pip install torch==2.1.2 torchvision==0.16.2  --index-url https://download.pytorch.org/whl/cu121

# Install other prequisites
pip install -r requirements.txt
# for other methods you should look for their requirements in their respective folders.
```

## 2. Annotation Tools
We put our annotation tool in [an independent GitHub repository](https://github.com/Lijiaxin0111/SemiAuto-Multi-Level-Annotation-Tool).

## 3. Evaluation
We only introduce the full precedure of reimplementing ReVOS_Cutie cause the others' are well written in their README files: ([Cutie-base](methods\Cutie\README.md),[RMem and DeAOT](methods\RMem\README.md), [SAM2](methods\segment-anything-2\README.md), [XMem](methods\XMem\README.md))

### 3.1 Download the model parameters
Download [Cutie_ReVOS parameters](https://drive.google.com/file/d/1fItxsooXXO0VVODyxYKVFRdtZrp-HANR/view?usp=sharing) and configure `weights` in `methods\Cutie_ReVOS\cutie\config\eval_config.yaml` or set `weights` as an argument in `methods\Cutie_ReVOS\eval.sh`

### 3.2. Download datasets
tip: we recommend you soft link all datasets under `./methods`, and configure the path for every method according to their requiements.
#### 3.2.1 M3VOS dataset
https://drive.google.com/drive/folders/1qNSvE6dpkCHSs_8eZRo6vruLScCHl7oI?usp=sharing
We unify the file structure of file like `VOST`:
```
- M3VOS
    	- JPEGIMages
      		- seq_0
        		- 0000000.jpg
        		...
      		- seq_1
      		...
    	-  Annotations
     		 - seq_0
        		- 0000000.png
       	 	    ...
      		 - seq_1
      		 ...
    	- Videos
      		-seq_0.mp4
      		-seq_1.mp4
    		...
	- ImageSets
		- val.txt
	- meta
	...
        
```
For Cutie_ReVOS, configure arguments of `roves-val` in `methods\Cutie_ReVOS\cutie\config\eval_config.yaml`

#### 3.2.2 other datasets
For VOST, DAVIS and YouTubeVOS, configure it as required in [EVALUATION of Cutie](methods\Cutie\docs\EVALUATION.md)

### 3.3 Run evaluation
For Cutie and Cutie_ReVOS, run evaluation as shown in [EVALUATION of Cutie](methods\Cutie\docs\EVALUATION.md)
For DeAOT, 

### 3.4 calculate scores
#### get  $J_st$ and $J$ for each instance and global average

```
#  modify the week_num in ./methods/evaluation/eval.sh 
# choose the result path of model
cd /methods
sbatch --gpus=1 evaluation/eval.sh 
```



#### get $J_st$ and $J$ for each challenge

```
#  modify the week_num in ./methods/evaluation/cal_challenge_score.sh 
# choose the result path of model
cd /methods
sbatch --gpus=1 evaluation/cal_challenge_score.sh 
```

:rocket: TIP: Three result `csv` will be store in the `result_path`

## 4. Training


## 5. Useful scripts

in the `./scripts`

- `merge_signle_video.py`:  merge the mask and image into a video  , args:
  - `images_folder`:  a folder contains images: `001.jpg`  , `002.jpg`, ...
  - `masks_folder`:  a folder contains masks: `001.png`  , `002.png`, ...
  - `output_video`： `*.mp4`
- `merge_png2video.py`: process the dataset whose file structure just like `VOST`, get the merge videos folder
  - `images_folder`:  a folder contains the images of seqs:  just like `ROVES_summary/ROVES_week_0/JPEGIMages `
  - `masks_folder`:  a folder contains he masks of  seqs: `ROVES_summary/ROVES_week_0/Annotations`
  - `output_video`：a target folder contains  merge videos : `exp/merge_videos`
- `align_direction.py`: If you find the width of your video is more than its height , it will rotate it 90 degree  counterclockwisely.



## predict segment

### AOT

```
#  modify the week_num in ./methods/RMem/aot_plus/eval_roves.sh
cd methods/RMem/aot_plus
sbatch --gpus=1 eval_roves.sh
```

- the prediction result in `methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_in_aot_week_${week_num}`



### DeAOT + Rmem

tips: the key parameter to decide whether use Rmem is `cfg.USE_TEMPORAL_POSITIONAL_EMBEDDING = True`

```
#  modify the week_num in ./methods/RMem/aot_plus/eval_roves_deaot.sh
cd methods/RMem/aot_plus
sbatch --gpus=1 eval_roves_deaot.sh
```

the prediction result in `methods/RMem/aot_plus/results/aotplus_R50_DeAOTL_Temp_pe_Slot_4/pre_vost/eval/roves/test_roves_week_in_deaotRmem_${week_num}`


### XMem
```
# modify the week_num in ./methods/XMem/eval.sh
cd methods/XMem
sbatch --gpus=1 -o eval.out eval.sh
```
the prediction result in `methods/XMem/output/roves_week${week_num}`

### Cutie
```
# modify the week_num in ./methods/Cutie/eval.sh
cd methods/Cutie
sbatch --gpus=1 -o eval.out eval.sh
```
the prediction result in `methods/Cutie/cutie_output/roves_week${week_num}`

### SAM2
```
# modify the week_num in ./methods/segment-anything-2/eval_roves.sh
cd methods/segment-anything-2

```
