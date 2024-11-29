import shutil, os 

datapath = '/home/jsi0613/projects/RFDiff-Sandbox/samples/errors/'
datalist = os.listdir(datapath) 

for data in datalist:
    pdb = datalist + data 
    with open(pdb, 'r') as f:
        lines = f.readlines()
        for l in lines:
            


