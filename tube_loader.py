
import h5py
import glob
import yt
import numpy as np
import pdb
import torch


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
                parameters[field]=float(chunk)
    output = [parameters[field] for field in fieldlist]
    return output


def consume_one(directory, frame):
    OutputLogName = "%s/OutputLog"%directory
    fptr = open(OutputLogName,'r')
    lines = fptr.readlines()
    pf1 = lines[frame].split()[2]
    fptr.close()

    ds = yt.load("%s/%s"%(directory,pf1))
    time = ds['InitialTime']

    fields = ['density','pressure','velocity_x']

    data = []
    for nf, field in enumerate(fields):
        ray = ds.ortho_ray(0,[0.0,0.0])
        data.append(list(ray[field].v))

    parameters = get_parameters(directory+"/tube.enzo")
    parameters.append(time)
    return data, parameters
def consume(directory):
    OutputLogName = "%s/OutputLog"%directory
    fptr = open(OutputLogName,'r')
    lines = fptr.readlines()
    pf0 = lines[0].split()[2]
    pf1 = lines[-1].split()[2]
    fptr.close()

    ds0 = yt.load("%s/%s"%(directory,pf0))
    ds1 = yt.load("%s/%s"%(directory,pf1))
    time = ds1['InitialTime']
    ds_list = [ds0,ds1]

    fields = ['density','pressure','velocity_x']

    pair = []
    for nds, ds in enumerate(ds_list):
        pair.append([])
        for nf, field in enumerate(fields):
            ray = ds.ortho_ray(0,[0.0,0.0])
            pair[nds].append(list(ray[field].v))

    parameters = get_parameters(directory+"/tube.enzo")
    parameters.append(time)
    return pair, parameters


def load_many(check_file=None):
    base = '/scratch3/dcollins/Paper79/tube_test/res1000/tubes'
    tubes = sorted(glob.glob("%s/tube*"%base))
    dataset = []
    N = len(tubes)
    numbers = []
    numbers_got = []
    parameters = []
    if check_file is not None:
        fptr = h5py.File(check_file,'r')
        numbers_got = fptr['numbers'][()]
        fptr.close()
    for nt,tube in enumerate(tubes):
        print("%s %d/%d"%(tube, nt, N))
        number = int(tube.split('/')[-1].split('_')[-1])
        if number in numbers_got:
            print('skip',number)
            continue
        numbers.append(number)
        boo= consume("%s"%(tube))
        dataset += [boo[0]]

        parameters.append(boo[1])
    dataset = np.array(dataset)
    return dataset, numbers, parameters

def load_many_time(check_file=None):
    base = '/scratch3/dcollins/Paper79/tube_test/unit/tubes'
    tubes = sorted(glob.glob("%s/tube*"%base))
    dataset = []
    N = len(tubes)
    numbers = []
    numbers_got = []
    parameters = []
    if check_file is not None:
        fptr = h5py.File(check_file,'r')
        numbers_got = fptr['numbers'][()]
        fptr.close()
    for nt,tube in enumerate(tubes):
        print("%s %d/%d"%(tube, nt, N))
        number = int(tube.split('/')[-1].split('_')[-1])
        if number in numbers_got:
            print('skip',number)
            continue
        numbers.append(number)
        for frame in range(1,12):
            boo= consume_one("%s"%(tube),frame)
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
    numbers = fptr['numbers'][()].astype('float')

    fptr.close()
    return tubes, parameters,numbers

nframes=11
def read_good_parameters_by_tube(fname, nvalid=100,ntest=100):
    tubes,parameters, numbers = read_all_parameters(fname)
    tubes_out=[]
    param_out=[]
    numbers_out=[]
    n=-1
    ntubes = len(tubes)//nframes
    for ntube in torch.arange(ntubes):
        keep=True
        this_tube=[]
        this_param=[]
        for frame in torch.arange(nframes):
            i = ntube*nframes+frame
            this_tube.append(tubes[i])
            this_param.append(parameters[i])

            f = tubes[i]
            param = parameters[i]
            for nf,field in enumerate(['density','pressure','velocity']):
                #print('L',f[nf][0:5], param[2*nf])
                if (np.abs(f[nf][0:20]-param[2*nf])>1e-5).any():
                    keep=False
                #print('R',f[nf][-5:], param[2*nf+1])
                if (np.abs(f[nf][-20:]-param[2*nf+1])>1e-5).any():
                    keep=False
        if keep:
            tubes_out += this_tube
            param_out += this_param
    #print("Start %d end %d"%(len(tubes), len(tubes_out)))

    alldata = torch.tensor(tubes_out,dtype=torch.float32)
    allparameters = torch.tensor(param_out,dtype=torch.float32)
    allnumbers = torch.tensor(numbers_out, dtype=torch.int)
    data={'validate':alldata[:nvalid], 'test':alldata[nvalid:nvalid+ntest],'train':alldata[nvalid+ntest:]}
    parameters={'validate':allparameters[:nvalid], 'test':allparameters[nvalid:nvalid+ntest],'train':allparameters[nvalid+ntest:]}
    #numbers={'validate':allnumbers[:nvalid], 'test':allnumbers[nvalid:nvalid+ntest],'train':allnumbers[nvalid+ntest:]}
    return data, parameters


def read_good_parameters(fname, nvalid=100,ntest=100):
    tubes,parameters, numbers = read_all_parameters(fname)
    tubes_out=[]
    param_out=[]
    numbers_out=[]
    n=-1
    for datum, param in zip(tubes,parameters):
        n+=1
        if len(datum.shape) == 3:
            f = datum[1]
        else:
            f = datum
        keep=True
        for nf,field in enumerate(['density','pressure','velocity']):
            #print('L',f[nf][0:5], param[2*nf])
            if (np.abs(f[nf][0:20]-param[2*nf])>1e-5).any():
                keep=False
            #print('R',f[nf][-5:], param[2*nf+1])
            if (np.abs(f[nf][-20:]-param[2*nf+1])>1e-5).any():
                keep=False
        if keep:
            tubes_out.append(datum)
            param_out.append(param)
            numbers_out.append(n//11)
    #print("Start %d end %d"%(len(tubes), len(tubes_out)))

    alldata = torch.tensor(tubes_out,dtype=torch.float32)
    allparameters = torch.tensor(param_out,dtype=torch.float32)
    allnumbers = torch.tensor(numbers_out, dtype=torch.int)
    data={'validate':alldata[:nvalid], 'test':alldata[nvalid:nvalid+ntest],'train':alldata[nvalid+ntest:]}
    parameters={'validate':allparameters[:nvalid], 'test':allparameters[nvalid:nvalid+ntest],'train':allparameters[nvalid+ntest:]}
    numbers={'validate':allnumbers[:nvalid], 'test':allnumbers[nvalid:nvalid+ntest],'train':allnumbers[nvalid+ntest:]}
    return data, parameters, numbers

def read_parameters(fname, nvalid=100,ntest=100):
    tubes_out,param_out, numbers_out = read_all_parameters(fname)

    alldata = torch.tensor(tubes_out,dtype=torch.float32)
    allparameters = torch.tensor(param_out,dtype=torch.float32)
    allnumbers = torch.tensor(numbers_out, dtype=torch.int)
    data={'validate':alldata[:nvalid], 'test':alldata[nvalid:nvalid+ntest],'train':alldata[nvalid+ntest:]}
    parameters={'validate':allparameters[:nvalid], 'test':allparameters[nvalid:nvalid+ntest],'train':allparameters[nvalid+ntest:]}
    numbers={'validate':allnumbers[:nvalid], 'test':allnumbers[nvalid:nvalid+ntest],'train':allnumbers[nvalid+ntest:]}
    return data, parameters, numbers

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

    


