import numpy as np
import os

from helpers import get_parent_path, makedir
from classification import struct_from_file, cells_and_masks_from_struct
from ProcessingPipeline import ProcessingPipeline

from colicoords import Cell,Data, data_to_cells
from tqdm import tqdm

#Load all untreated cells

data_folder = r'C:\Users\zagajewski\PycharmProjects\AMR\Data\Exp1'

output_folder = os.path.join(get_parent_path(1), 'Data', 'Nucleoid Reconstruction')

makedir(output_folder)

output_segregated = os.path.join(output_folder, 'Segregated_Multichannel')
output_collected = os.path.join(output_folder, 'Collected_Multichannel')
output_dataset = os.path.join(output_folder, 'Dataset_Multichannel')


cond_IDs = ['WT+ETOH']
image_channels = ['NR', 'DAPI']
img_dims = ((30,1), 684, 840)

input_path_WT = os.path.join(get_parent_path(1), 'Data', 'Segmentations_All', 'WT+ETOH')

pipeline = ProcessingPipeline(data_folder, 'NIM')
#pipeline.Sort(cond_IDs=cond_IDs, img_dims=img_dims, image_channels=image_channels,
 #             crop_mapping={'DAPI': 0, 'NR': 0}, output_folder=output_segregated)
#pipeline.Collect(cond_IDs=cond_IDs, image_channels=image_channels, output_folder=output_collected,
  #               registration_target=0)

# --- RETRIEVE MASKS AND MATCHING FILES, SPLIT INTO SETS INTO ONE DATABASE---
annots_WT = os.path.join(input_path_WT, 'annots')
files_WT = os.path.join(output_collected, 'WT+ETOH')


#pipeline.FileOp('TrainTestVal_split', data_sources=[files_WT],
 #               annotation_sources=[annots_WT], output_folder=output_dataset, test_size=0.2,
  #              validation_size=0.2, seed=42)

manual_struct = struct_from_file(dataset_folder=output_dataset,
                                 class_id=1)

cells = cells_and_masks_from_struct(input=manual_struct, cond_IDs=cond_IDs, image_dir=output_collected, mode='masks')['WT+ETOH']

#Build list of cells
cellcount = len(cells)
celllist = [0]*cellcount

for i in tqdm(range(cellcount),desc='Loading and rotating cells.'):

    cell_input = cells[i]
    mask = cell_input[1]
    NR = cell_input[0][:,:,0]
    DAPI = cell_input[0][:,:,1]


    data = Data()
    data.add_data(np.expand_dims(mask,0), 'binary')
    data.add_data(np.expand_dims(NR,0), 'fluorescence', name='NR')
    data.add_data(np.expand_dims(DAPI,0), 'fluorescence', name='DAPI')

    #Rotate cell horizontally
    cell = data_to_cells(data, remove_bordering=False, remove_multiple_cells=False)[0]

    celllist[i] = cell



