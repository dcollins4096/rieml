
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
def read_all_parameters(fname):
    fptr=h5py.File(fname,'r')
    tubes = fptr['tubes'][()]
    parameters = fptr['parameters'][()].astype('float')

    fptr.close()
    return tubes, parameters

def read_good_parameters(fname):
    tubes,parameters = read_all_parameters(fname)
    tubes_out=[]
    param_out=[]
    for datum, param in zip(tubes,parameters):
        f = datum[1]
        keep=True
        for nf,field in enumerate(['density','pressure','velocity']):
            #print('L',f[nf][0:5], param[2*nf])
            if (np.abs(f[nf][0:10]-param[2*nf])>1e-5).any():
                keep=False
            #print('R',f[nf][-5:], param[2*nf+1])
            if (np.abs(f[nf][-10:]-param[2*nf+1])>1e-5).any():
                keep=False
        if keep:
            tubes_out.append(datum)
            param_out.append(param)
    #print("Start %d end %d"%(len(tubes), len(tubes_out)))
    return tubes_out,param_out

def extract_validation(model, data, parameters):
    tubes_out=[]
    param_out=[]
    ind_out = []
    ind = list(range(len(data)))
    n=-1
    for datum, param in zip(data,parameters):
        n+=1
        f = datum[1]
        keep=True
        for nf,field in enumerate(['density','pressure','velocity']):
            #print('L',f[nf][0:5], param[2*nf])
            if (np.abs(f[nf][0:10]-param[2*nf])>1e-5).any():
                keep=False
            #print('R',f[nf][-5:], param[2*nf+1])
            if (np.abs(f[nf][-10:]-param[2*nf+1])>1e-5).any():
                keep=False
        if keep:
            tubes_out.append(datum)
            param_out.append(param)
            ind_out.append(ind[n])

    models = [model(param.view(1,6)) for param in param_out]
    losses = torch.tensor([model.criterion1(mod.view(1,3,1000), dat[1].view(1,3,1000)) for mod,dat in zip(models_test,tubes_out)])
    arg = np.argsort(losses)
    N = len(ind_out)//2
    ind_validate = ind_out[:5]+ ind_out[-5:] + ind_out[N:N+5]
    return ind_validate

    


