import os

datapath = r'C:\Users\zagajewski\Desktop\ABX_trial\Trial2'
NEWDATE = '220816'

for root, dirs, files in os.walk(datapath):
    for file in files:

        if not file.endswith('.tif'):
            continue

        source = os.path.join(root,file)

        [file,_] = file.split('.')
        [A,ExpID,ProtID,_,UserID,StrainID,CONDID,ALL_CHANNELS,CHANNELSERIES,POSXY,_,_,POSZ] = file.split('_')

        try:
            CONDID,CONC = CONDID.split('@')
            CONC = '[' + CONC + ']'
        except ValueError:
            CONDID = CONDID
            CONC = 'NA'



        fname = NEWDATE+'_1_1_AMR_' + UserID + '_' + StrainID + '_' + CONDID + '_' + 'DAPI+NR' + '_' + CONC + '_' + CHANNELSERIES + '_' + POSXY + '_channels' + '_t0' + '_posZ0.tif'

        dest = os.path.join(root,fname)

        os.rename(source,dest)
