# Evaluation
We only introduce the full precedure of reimplementing ReVOS_Cutie cause the others' are well written in their README files: ([Cutie-base](../methods/Cutie/README.md),[RMem and DeAOT](../methods/RMem/README.md), [SAM2](../methods/segment-anything-2/README.md), [XMem](../methods/XMem/README.md))

### 1. Download the model parameters
Download [Cutie_ReVOS parameters](https://drive.google.com/file/d/1fItxsooXXO0VVODyxYKVFRdtZrp-HANR/view?usp=sharing) and configure `weights` in `methods\Cutie_ReVOS\cutie\config\eval_config.yaml` or set `weights` as an argument in `methods\Cutie_ReVOS\eval.sh`

### 2. Download datasets
tip: we recommend you soft link all datasets under `./methods`, and configure the path for every method according to their requiements.
#### 2.1 M$^3$VOS dataset
download M$^3$VOS dataset at [this link](https://drive.google.com/drive/folders/1qNSvE6dpkCHSs_8eZRo6vruLScCHl7oI?usp=sharing).
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
For VOST, DAVIS and YouTubeVOS, download and configure it as required in [EVALUATION of Cutie](../methods/Cutie/docs/EVALUATION.md)

### 3. Run evaluation
For Cutie and Cutie_ReVOS, run evaluation as shown in [EVALUATION of Cutie](../methods/Cutie/docs/EVALUATION.md)

For DeAOT, run [this file]()

For RMem+DeAOT, run [this file]()

For XMem, run [this file](../methods/XMem/eval.sh)

For SAM2, run [this file](../methods/segment-anything-2/eval.sh)

### 4. calculate scores
First, you need set dataset path on `methods\evaluation\evaluation_method.py`

For REVOS, configure and run [this file](../methods/evaluation/eval_roves.sh)

For VOST, configure and run [this file](../methods/evaluation/eval_vost.sh)

For DAVIS, configure and run [this file](../methods/evaluation/eval_davis2017.sh)

For YouTube, you need to submit the output file to [online evaluation page]().