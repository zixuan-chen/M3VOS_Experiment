import matplotlib.pyplot as plt
import seaborn as sns
from cutie.inference.inference_core import InferenceCore
from torch.utils.data import DataLoader

roves_week8 = "/home/bingxing2/home/scx8ah2/dataset/ROVES_summary/ROVES_week_8"
vid_name = "0230_crystallize_liquid_7"
# 假设attention_weights是一个二维数组，包含了注意力权重



# 创建一个heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(attention_weights, annot=True, fmt=".2f", cmap="RdBu_r")
plt.title("Attention Weights")
plt.show()