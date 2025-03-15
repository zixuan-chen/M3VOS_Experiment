# DeformVOS



## datasets prepare 

We unify the file structure of file like `VOST` , as follow in `./datasets`:

```
- ROVES_summary
	- ROVES_week_0 
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
		-challenge_label.json
	...
        
        
- VOST (just_like ROVES_week_0)

- ...
```

tip: This folder `datasets` should be linked in the every method folder in  `./methods` 

​	for example, in CFBI， `datasets -> ../../datasets/`

​	In order to share the datasets in different which make it convenient to manager the datasets



## tool introduction

in the `./tool`

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

### CFBI

```
# modify the week_num in ./method/CFBI/roves_eval_fast.sh
cd ./methods/CFBI
sbatch --gpus=1 roves_eval_fast.sh 
```

- the prediction result in `methods/CFBI/result/resnet101_cfbi/eval/roves/roves_test_roves_week_{week_num}_ckpt_unknown`



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
sbatch -x paraai-n32-h-01-agent-[1,4,7-8,16-17]    --gpus=1 -o eval.out eval_roves.sh
```

## Eval code

### get  $J_st$ and $J$ for each instance and global average

```
#  modify the week_num in ./methods/evaluation/eval.sh 
# choose the result path of model
cd /methods
sbatch --gpus=1 evaluation/eval.sh 
```



### get $J_st$ and $J$ for each challenge

```
#  modify the week_num in ./methods/evaluation/cal_challenge_score.sh 
# choose the result path of model
cd /methods
sbatch --gpus=1 evaluation/cal_challenge_score.sh 
```



:rocket: TIP: Three result `csv` will be store in the `result_path`


### Merge Video

In A100, using the `LLMSeg_cp310` conda env