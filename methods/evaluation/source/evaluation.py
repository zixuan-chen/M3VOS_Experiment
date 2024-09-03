import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from source.dataset import Dataset
from source.metrics import db_eval_boundary, db_eval_iou
from  source.metrics import db_eval_blob_torch as db_eval_blob
from source import utils
from source.results import Results
from scipy.optimize import linear_sum_assignment
from math import floor
import multiprocessing as mp
from multiprocessing import Manager


class Evaluation(object):
    def __init__(self, dataset_root, gt_set, sequences='all', fps=24):
        """
        Class to evaluate sequences from a certain set
        :param dataset_root: Path to the dataset folder that contains JPEGImages, Annotations, etc. folders.
        :param gt_set: Set to compute the evaluation
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.dataset_root = dataset_root
        print(f"Evaluate on dataset = {self.dataset_root}")        
        self.dataset = Dataset(root=dataset_root, subset=gt_set, sequences=sequences)
        self.compress_ratio = int(24 / fps)

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
        if all_res_masks.shape[0] > all_gt_masks.shape[0]:
            print("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            all_res_masks = all_res_masks[:all_gt_masks.shape[0]]
            # sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res = np.zeros(all_gt_masks.shape[:2])
        blob_metrics_res = np.zeros_like(j_metrics_res)
        

        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            if 'J_cc' in metric:
                blob_metrics_res[ii, :] = db_eval_blob(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)

        return j_metrics_res , blob_metrics_res

    def evaluate(self, res_path, metric=('J', 'J_last', "J_cc"), debug=False):
        
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        cnt = 0

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'J_last' in metric:
            metrics_res['J_last'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        if 'J_cc' in metric:
            metrics_res["J_cc"] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        process_list = []
        sema = mp.Semaphore(1)
        manager = Manager()
        metrics_res = manager.dict({
            'J': manager.dict(
                {
                    "M": manager.list(),
                    "R": manager.list(),
                    "D": manager.list(),
                    "M_per_object": manager.dict(),
                }
            ),
            'J_last': manager.dict(
                {
                    "M": manager.list(),
                    "R": manager.list(),
                    "D": manager.list(),
                    "M_per_object": manager.dict(),
                }
            ),
            'J_cc': manager.dict(
                {
                    "M": manager.list(),
                    "M_per_object": manager.dict(),
                }    
            )
        })
        # Sweep all sequences
        results = Results(root_dir=res_path)
        for seq in tqdm(list(self.dataset.get_sequences())):
        # sequences = list(results.get_sequences())
        # for seq in tqdm(sequences):
            def evaluate():
                print(f"\n{seq}")
                try:
                    all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
                    print(f"\n{seq} DONE 1")
                    num_objects = all_gt_masks.shape[0]
                    print(f"\n{seq} DONE 1.4")
                    all_gt_masks = all_gt_masks[:, : :  self.compress_ratio]
                    print(f"\n{seq} DONE 1.5")
                    all_masks_id = all_masks_id[: : self.compress_ratio]
                    print(f"\n{seq} DONE 1.6")
                    all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
                    print(f"\n{seq} DONE 1.7")
                    num_eval_frames = len(all_masks_id)
                    print(f"\n{seq} DONE 1.8")
                    last_quarter_ind = int(floor(num_eval_frames * 0.75))
                    print(f"\n{seq} DONE 1.9")
        

                    all_res_masks = results.read_masks(seq, all_masks_id)
                    print(f"\n{seq} DONE 2")
                    j_metrics_res, blob_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
                    print(f"\n{seq} DONE 3")
        

                    for ii in range(all_gt_masks.shape[0]):
                        seq_name = f'{seq}_{ii+1}'
                        if 'J' in metric:
                            [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                            metrics_res['J']["M"].append(JM)
                            metrics_res['J']["R"].append(JR)
                            metrics_res['J']["D"].append(JD)
                            metrics_res['J']["M_per_object"][seq_name] = JM
                        if 'J_last' in metric:
                            [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii][last_quarter_ind:])
                            metrics_res['J_last']["M"].append(JM)
                            metrics_res['J_last']["R"].append(JR)
                            metrics_res['J_last']["D"].append(JD)
                            metrics_res['J_last']["M_per_object"][seq_name] = JM
                        if 'J_cc' in metric:
                            metrics_res['J_cc']["M"].append(np.mean(blob_metrics_res[ii]))
                            metrics_res['J_cc']["M_per_object"][seq_name] = np.mean(blob_metrics_res[ii])

                    print(f"\n{seq} DONE 4")

                    # Show progress
                    if debug:
                        sys.stdout.write(seq + '\n')
                        sys.stdout.flush()
                except Exception as e:
                    print(f"Error in evaluate: {e}")
                finally:
                    sema.release()
                    print(f"{seq} complete ! ")
                    
            sema.acquire()
            p = mp.Process(target=evaluate, args=())
            p.start()
            process_list.append(p)
        for process in tqdm(process_list):
            process.join()
            cnt += 1
            print("waitting:" , cnt)
            
        return metrics_res

