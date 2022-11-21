
from ProcessingPipeline import ProcessingPipeline
import os
from helpers import *

from classification import *

cond_IDs = ['WT+ETOH', 'RIF+ETOH', 'CIP+ETOH']
image_channels = ['NR','DAPI']
img_dims = (30,684,840)


output_collected = os.path.join(get_parent_path(1),'Data','Exp1_Collected_Multichannel')
manual_struct = struct_from_file(dataset_folder=os.path.join(get_parent_path(1), 'Data', 'Dataset_Exp1_Multichannel'),
                                 class_id=1)

cells = cells_from_struct(input=manual_struct, cond_IDs=cond_IDs, image_dir=output_collected, mode='masks')
X_train, X_test, y_train, y_test = split_cell_sets(input=cells, test_size=0.2, random_state=42)

#Fix pycharm console
class PseudoTTY(object):
    def __init__(self, underlying):
        self.__underlying = underlying
    def __getattr__(self, name):
        return getattr(self.__underlying, name)
    def isatty(self):
        return True

sys.stdout = PseudoTTY(sys.stdout)

logdir = os.path.join(get_parent_path(1), 'Second_Stage_2')
resize_target = (64, 64, 3)
class_count = 3

inspect(modelpath=os.path.join(logdir,'DenseNet121_EXP1.h5'), X_test=X_test, y_test=y_test, mean=np.asarray([0, 0, 0]), resize_target=resize_target,
        class_id_to_name=cells['class_id_to_name'])