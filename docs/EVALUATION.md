# Evaluation
We only introduce the full precedure of reimplementing ReVOS_Cutie cause the others' are well written in their README files: ([Cutie-base](../methods/Cutie/README.md),[RMem and DeAOT](../methods/RMem/README.md), [SAM2](../methods/segment-anything-2/README.md), [XMem](../methods/XMem/README.md))

### 1. Download the model parameters
Download [Cutie_ReVOS parameters](https://drive.google.com/file/d/1fItxsooXXO0VVODyxYKVFRdtZrp-HANR/view?usp=sharing) and configure `weights` in `methods\Cutie_ReVOS\cutie\config\eval_config.yaml` or set `weights` as an argument in `methods\Cutie_ReVOS\eval.sh`

### 2. Download datasets
tip: we recommend you soft link all datasets under `./methods`, and configure the path for every method according to their requiements.
#### 2.1 M$^3$VOS dataset
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

#### 2.2 other datasets
For VOST, DAVIS and YouTubeVOS, configure it as required in [EVALUATION of Cutie](../methods/Cutie/docs/EVALUATION.md)

### 3. Run evaluation
For Cutie and Cutie_ReVOS, run evaluation as shown in [EVALUATION of Cutie](../methods/Cutie/docs/EVALUATION.md)
For DeAOT, 

### 4. calculate scores
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