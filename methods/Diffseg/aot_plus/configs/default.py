import os
import importlib


class DefaultEngineConfig():
    def __init__(self, exp_name='default', model='AOTT'):
        model_cfg = importlib.import_module('configs.models.' +
                                            model).ModelConfig()
        self.__dict__.update(model_cfg.__dict__)  # add model config

        self.EXP_NAME = exp_name + '_' + self.MODEL_NAME

        self.STAGE_NAME = 'default'

        # Dataset --------------------------

        self.DATASETS = ['youtubevos']
        self.DATA_WORKERS = 8
        self.DATA_RANDOMCROP = (465,
                                465) if self.MODEL_ALIGN_CORNERS else (464,
                                                                       464) # 随机采样视频中的一个块块作为训练图片
        self.DATA_RANDOMFLIP = 0.5
        self.DATA_MAX_CROP_STEPS = 10 
        self.DATA_SHORT_EDGE_LEN = 480 # 先把较短变缩放到一定程度，然后再进一步进行随机缩放
        self.DATA_MIN_SCALE_FACTOR = 0.7 # 最小的缩放因子
        self.DATA_MAX_SCALE_FACTOR = 1.3 # 最大的缩放因子
        self.DATA_RANDOM_REVERSE_SEQ = True # 以一定概率随机逆反视频
        self.DATA_SEQ_LEN = 5 # 每个样本的序列长度
        self.DATA_DAVIS_REPEAT = 5 # 每一帧的重复次数 for DAVIS
        self.DATA_VOST_REPEAT = 1 # 每一帧的重复次数 for VOST
        self.DATA_VOST_IGNORE_THRESH = 0.2  # 样本里面的无效区域占有效区域的比例不超过
        self.DATA_VOST_ALL_FRAMES = False
        self.DATA_VOST_VALID_FRAMES = False
        self.DATA_RANDOM_GAP_DAVIS = 12  # max frame interval between two sampled frames for DAVIS (24fps)
        self.DATA_RANDOM_GAP_YTB = 3  # max frame interval between two sampled frames for YouTube-VOS (6fps)
        self.DATA_RANDOM_GAP_VOST = 3 # max frame interval between two sampled frames for VOST
        self.DATA_RANDOM_GAP_VISOR = 1  # max frame interval between two sampled frames for VISOR
        self.DATA_DYNAMIC_MERGE_PROB = 0.2  # 随机合并两个样本的概率
        self.IGNORE_IN_MERGE = True # 合并后去掉两个样本的void region
        self.DATA_VISOR_REPEAT = 1 
        self.DATA_VISOR_IGNORE_THRESH = 0.2
        # --------------------------


        # Pretrain 
        self.PRETRAIN = True
        self.PRETRAIN_FULL = False  # if False, load encoder only
        self.PRETRAIN_MODEL = ''
        # ----------------

        # train 
        self.TRAIN_TOTAL_STEPS = 100000 # 总共的训练轮次
        self.TRAIN_START_STEP = 0  # 导入预训练模型后的起始训练step
        self.TRAIN_WEIGHT_DECAY = 0.07
        self.TRAIN_WEIGHT_DECAY_EXCLUSIVE = {
            # 'encoder.': 0.01
        }
        self.TRAIN_WEIGHT_DECAY_EXEMPTION = [
            'absolute_pos_embed', 'relative_position_bias_table',
            'relative_emb_v', 'conv_out'
        ]
        self.TRAIN_LR = 2e-4
        self.TRAIN_LR_MIN = 2e-5 if 'mobilenetv2' in self.MODEL_ENCODER else 1e-5
        self.TRAIN_LR_POWER = 0.9
        self.TRAIN_LR_ENCODER_RATIO = 0.1 # encoder的学习率：(now_lr - min_lr) * encoder_lr_ratio + min_lr
        self.TRAIN_LR_WARM_UP_RATIO = 0.05 # warm up的学习过程占比
        self.TRAIN_LR_COSINE_DECAY = False
        self.TRAIN_LR_RESTART = 1 # 如果！= 1, 每restart 个step,就重新开始训练过程
        self.TRAIN_LR_UPDATE_STEP = 1 
        self.TRAIN_AUX_LOSS_WEIGHT = 1.0  #  
        self.TRAIN_AUX_LOSS_RATIO = 1.0
        self.TRAIN_OPT = 'adamw'
        self.TRAIN_SGD_MOMENTUM = 0.9
        self.TRAIN_GPUS = 4
        self.TRAIN_BATCH_SIZE = 16
        self.TRAIN_TBLOG = False
        self.TRAIN_TBLOG_STEP = 50 # tensorboard log frequence
        self.TRAIN_LOG_STEP = 20
        self.TRAIN_IMG_LOG = True
        self.TRAIN_TOP_K_PERCENT_PIXELS = 0.15 # 在计算cross entropy的时候, 只计算loss最大的像素
        self.TRAIN_SEQ_TRAINING_FREEZE_PARAMS = ['patch_wise_id_bank'] # step >= start_seq_training_steps时,不再训练 freeze的模型参数
        self.TRAIN_SEQ_TRAINING_START_RATIO = 0.5
        self.TRAIN_HARD_MINING_RATIO = 0.5 # 训练前半阶段,计算尽可能多的像素,随着step推移,逐渐接近topk
        self.TRAIN_EMA_RATIO = 0.1
        self.TRAIN_CLIP_GRAD_NORM = 5.
        self.TRAIN_SAVE_STEP = 500
        self.TRAIN_EVAL = False
        self.TRAIN_MAX_KEEP_CKPT = 8
        self.TRAIN_RESUME = False
        self.TRAIN_RESUME_CKPT = None
        self.TRAIN_RESUME_STEP = 0
        self.TRAIN_AUTO_RESUME = True
        self.TRAIN_DATASET_FULL_RESOLUTION = False
        self.TRAIN_ENCODER_FREEZE_AT = 2
        try:
            self.TRAIN_ENCODER_FREEZE_AT = 4 if self.TOP_DOWN_FREEZE_ENCODER else self.TRAIN_ENCODER_FREEZE_AT
        except AttributeError:
            pass
        self.TRAIN_LSTT_EMB_DROPOUT = 0.
        self.TRAIN_LSTT_ID_DROPOUT = 0.
        self.TRAIN_LSTT_DROPPATH = 0.1
        self.TRAIN_LSTT_DROPPATH_SCALING = False
        self.TRAIN_LSTT_DROPPATH_LST = False
        self.TRAIN_LSTT_LT_DROPOUT = 0.
        self.TRAIN_LSTT_ST_DROPOUT = 0.

        self.TEST_GPU_ID = 0
        self.TEST_GPU_NUM = 1
        self.TEST_FRAME_LOG = False
        self.TEST_DATASET = 'youtubevos'
        self.TEST_DATASET_FULL_RESOLUTION = False
        self.TEST_DATASET_SPLIT = 'val'
        self.TEST_CKPT_PATH = None
        # if "None", evaluate the latest checkpoint.
        self.TEST_CKPT_STEP = None
        self.TEST_FLIP = False
        self.TEST_MULTISCALE = [1]
        self.TEST_MIN_SIZE = None
        self.TEST_MAX_SIZE = 800 * 1.3
        self.TEST_WORKERS = 4

        # GPU distribution
        self.DIST_ENABLE = True
        self.DIST_BACKEND = "gloo"
        self.DIST_URL = "tcp://127.0.0.1:13241"
        self.DIST_START_GPU = 0

    def init_dir(self):
        self.DIR_DATA = './datasets'
        self.DIR_DAVIS = os.path.join(self.DIR_DATA, 'DAVIS')
        self.DIR_LONG_VIDEOS = os.path.join(self.DIR_DATA, 'long_videos')
        self.DIR_VOST = os.path.join(self.DIR_DATA, 'VOST')
        self.DIR_VISOR = os.path.join(self.DIR_DATA, 'VISOR')
        self.DIR_YTB = os.path.join(self.DIR_DATA, 'YTB')
        self.DIR_STATIC = os.path.join(self.DIR_DATA, 'Static')

        self.DIR_ROOT = './results'

        self.DIR_RESULT = os.path.join(self.DIR_ROOT, self.EXP_NAME,
                                       self.STAGE_NAME)
        self.DIR_CKPT = os.path.join(self.DIR_RESULT, 'ckpt')
        self.DIR_EMA_CKPT = os.path.join(self.DIR_RESULT, 'ema_ckpt')
        self.DIR_LOG = os.path.join(self.DIR_RESULT, 'log')
        self.DIR_TB_LOG = os.path.join(self.DIR_RESULT, 'log', 'tensorboard')
        self.DIR_IMG_LOG = os.path.join(self.DIR_RESULT, 'log', 'img')
        self.DIR_EVALUATION = os.path.join(self.DIR_RESULT, 'eval')

        for path in [
                self.DIR_RESULT, self.DIR_CKPT, self.DIR_EMA_CKPT,
                self.DIR_LOG, self.DIR_EVALUATION, self.DIR_IMG_LOG,
                self.DIR_TB_LOG
        ]:
            if not os.path.isdir(path):
                try:
                    os.makedirs(path)
                except Exception as inst:
                    print(inst)
                    print('Failed to make dir: {}.'.format(path))

    def save_self(self):
        with open(os.path.join(self.DIR_RESULT, "config.py"), 'w') as f:
            f.write("""
class Config():
    def __init__(self):
""")
            for k, v in self.__dict__.items():
                v = f'"{v}"' if isinstance(v, str) else v
                f.write(f"""\
        self.{k} = {v}
""")

