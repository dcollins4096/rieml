
import h5py
import glob
import yt
import numpy as np
import pdb

def get_parameters(fname):
    fptr = open(fname,'r')
    lines = fptr.readlines()
    fptr.close()
    parameters = {}
    fieldlist = ["MHDBlastDA", "MHDBlastDB",
                 "MHDBlastPA", "MHDBlastPB" ,
                 "MHDBlastVelocityA", "MHDBlastVelocityB"
                ]
                 #"MHDBlastBA", "MHDBlastBB", 


    for line in lines:
        for field in fieldlist:
            if line.startswith(field):
                #only want the first velocity
                chunk = line.split('=')[1].strip().split(" ")[0]
                parameters[field]=chunk
    output = [parameters[field] for field in fieldlist]
    return output


def consume(directory):
    OutputLogName = "%s/OutputLog"%directory
    fptr = open(OutputLogName,'r')
    lines = fptr.readlines()
    pf0 = lines[0].split()[2]
    pf1 = lines[-1].split()[2]
    fptr.close()

    ds0 = yt.load("%s/%s"%(directory,pf0))
    ds1 = yt.load("%s/%s"%(directory,pf1))
    ds_list = [ds0,ds1]

    fields = ['density','pressure','velocity_x']

    pair = []
    for nds, ds in enumerate(ds_list):
        pair.append([])
        for nf, field in enumerate(fields):
            ray = ds.ortho_ray(0,[0.0,0.0])
            pair[nds].append(list(ray[field].v))

    parameters = get_parameters(directory+"/tube.enzo")
    return pair, parameters


def load_many(check_file=None):
    base = '/scratch3/dcollins/Paper79/tube_test/tubes'
    tubes = sorted(glob.glob("%s/tube*"%base))[:100]
    dataset = []
    N = len(tubes)
    numbers = []
    numbers_got = []
    parameters = []
    if check_file is not None:
        fptr = h5py.File(check_file,'r')
        numbers_got = fptr['numbers'][()]
    for nt,tube in enumerate(tubes):
        print("%s %d/%d"%(tube, nt, N))
        number = int(tube.split('/')[-1].split('_')[-1])
        if number in numbers_got:
            print('skip')
            continue
        numbers.append(number)
        boo= consume("%s"%(tube))
        dataset += [boo[0]]

        parameters.append(boo[1])
    dataset = np.array(dataset)
    return dataset, numbers, parameters

def write_one(fname, data):
    fptr=h5py.File(fname,'w')
    fptr.create_dataset('tubes',data=data[0])
    fptr.create_dataset('numbers',data=data[1])
    fptr.create_dataset('parameters',data=data[2])
    fptr.close()
def read_one(fname):
    fptr=h5py.File(fname,'r')
    tubes = fptr['tubes'][()]
    fptr.close()
    return tubes
def read_one_parameters(fname):
    fptr=h5py.File(fname,'r')
    tubes = fptr['tubes'][()]
    parameters = fptr['parameters'][()].astype('float')

    fptr.close()
    return tubes, parameters
