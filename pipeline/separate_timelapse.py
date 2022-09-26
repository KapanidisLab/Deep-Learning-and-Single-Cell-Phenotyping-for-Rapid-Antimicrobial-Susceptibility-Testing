import os,distutils.file_util as util

import numpy as np

from helpers import makedir
import skimage.io as io
import numpy
import skimage.exposure
from helpers import im_2_uint16

def autocontrast(img):
    v_min, v_max = numpy.percentile(img, (0.1, 99.9))
    img = skimage.exposure.rescale_intensity(img, in_range=(v_min, v_max))
    return img






datapath = r'C:\Users\zagajewski\Desktop\Tlapse\20220724_HUGFP_CEFT_2min_WGA647+GFP\WGA647_GFP'
output = r'C:\Users\zagajewski\Desktop\Tlapse\20220724_HUGFP_CEFT_2min_WGA647+GFP\WGA647_GFP\analysed'
correct_contrast = True

mapping = {'green': (1,'left'), 'red': (0,'right')} # colour to map to : (frame index , FOV side to crop)

out = []
times = []
makedir(output)



for root, dirs, files in os.walk(datapath):
    for file in files:
        if not file.endswith('.tif'):
            continue

        readpath = os.path.join(root,file)
        print(readpath)
        img = io.imread(readpath)

        if len(img) != 0:
            img_sz,img_sy,img_sx = img.shape

            newimg = np.zeros((3,img_sy,int(img_sx/2)))

            assert 'green' in list(mapping.keys()) and 'red' in list(mapping.keys())

            for key in mapping.keys():
                frame,side = mapping[key]

                i = img[frame,:,:]

                if side == 'left':
                    i = i[:,0:int(img_sx / int(2))]  # Crop FoV to remove unused half - keep left side
                elif side == 'right':
                    i = i[:,int(img_sx / int(2)):]  # Crop FoV to remove unused half - keep right side


                i = autocontrast(i) if correct_contrast else i  # Adjust contrast
                i = im_2_uint16(i)

                if key is 'red': insert_idx = 0
                if key is 'green': insert_idx = 1

                newimg[insert_idx,:,:] = i

            i = newimg

        else:
            i = img
            i = autocontrast(i) if correct_contrast else i  # Adjust contrast
            i = im_2_uint16(i)


        out.append(i)
        [_,_,_,_,_,_,_,_,t,_]=file.split('_')
        assert t[0] == 't'
        timepoint = t[1:]
        times.append(int(timepoint))


sorted_idx = numpy.asarray(times).argsort()
sorted_out = [out[i] for i in sorted_idx]
writepath = os.path.join(output,'stack.tiff')
w = numpy.asarray(sorted_out)
io.imsave(writepath,w)