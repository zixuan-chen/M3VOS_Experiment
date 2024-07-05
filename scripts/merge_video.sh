imgs="/home/lijiaxin/Deform_VOS/DeformVOS/data/VOST/JPEGImages_10fps"
masks="/home/lijiaxin/Deform_VOS/DeformVOS/tmp/RMem/aot_plus/results/aotplus_R50_AOTL_Temp_pe_Slot_4/pre_vost/eval/vost/debug"
exp="/home/lijiaxin/Deform_VOS/DeformVOS/exp/RMEM_video_10fps"

python tool/merge_png2mp4.py --images_folder "$imgs" --output_exp "$exp" --masks_folder "$masks"