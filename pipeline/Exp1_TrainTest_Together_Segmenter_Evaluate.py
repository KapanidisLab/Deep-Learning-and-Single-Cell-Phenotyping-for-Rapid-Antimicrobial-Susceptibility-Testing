import os
from ProcessingPipeline import ProcessingPipeline
import os
from helpers import *
from helpers import *
from implementations import *
from mask_generators import *
from segmentation import *
from classification import *

from classification import *

configuration = BacConfig()
configuration.NAME = 'FirstStage1'

weights = 'adasd'
test_dir = os.path.join(get_parent_path(1), 'Data', 'Dataset_Exp1', 'Test')


evaluate_coco_metrics(dataset_folder=test_dir, config=configuration,
                      weights=os.path.join(get_parent_path(1), "firststage120211105T0215",
                                           "mask_rcnn_EXP1.h5"))

inspect_mrcnn_segmenter(test_folder = test_dir, configuration = configuration, weights = weights, ids=[0,20,30])