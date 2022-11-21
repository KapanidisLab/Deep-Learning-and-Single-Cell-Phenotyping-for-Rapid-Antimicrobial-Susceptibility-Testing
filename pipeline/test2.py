import os

datapath= r'C:\Users\zagajewski\PycharmProjects\AMR\Data\Conor_DFP_Segmentations_all'

for root, dirs, files in os.walk(datapath):
    for file in files:

        if not file.endswith('.tif'):
            continue

        source = os.path.join(root,file)

        [file,_] = file.split('.')
        [DATE,EXPID,PRID,ProjectCode,CONC,UserID,StrainID,CONDID,CHANNELS,CHANNELSERIES,posXY,posZ] = file.split('_')

        if 'NA' in CONDID:
            print('Renaming {}'.format(file))
            CONDID = 'WT+ETOH'

            fname = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.tif'.format(DATE,EXPID,PRID,ProjectCode,CONC,UserID,StrainID,CONDID,CHANNELS,CHANNELSERIES,posXY,posZ)

            dest = os.path.join(root,fname)

            os.rename(source,dest)
