
import jinja2
import numpy as np
import glob
import os
import shutil
import yt
import matplotlib.pyplot as plt
import numpy as np
import pdb



stuff={}
def randy(low,high):
    x = np.random.random()
    d = high-low
    return x*d+low

def runner(directory):
    os.system("cd %s; enzo.exe -d tube.enzo > output 2>&1"%directory)
    shutil.move(directory,"tubes/%s"%directory)
def maker():
    stuff['gamma']=1.66666667
    stuff['density_left']   = 1
    stuff['density_right']  = randy(1e-3,2)
    stuff['pressure_left']  = 1/stuff['gamma'] #so cs = sqrt(gamma p/rho) = 1
    stuff['pressure_right'] = randy(1e-3,2)
    stuff['velocity_left']  = 0
    stuff['velocity_right'] = randy(-1,1)
    #mean_velocity = 0.5*(stuff['velocity_left']+stuff['velocity_right'])
    #stuff['velocity_left'] -= mean_velocity
    #stuff['velocity_right'] -= mean_velocity


    nruns = len(glob.glob("tubes/tube*"))
    print("Nruns",nruns)
    name = "tube_%05d"%(nruns+1)
    os.mkdir(name)

    fname = "%s/tube.enzo"%name
    loader=jinja2.FileSystemLoader('.')
    env = jinja2.Environment(loader=loader)
    template = env.get_template('template.enzo')
    foutptr = open(fname,'w')
    foutptr.write( template.render(**stuff))
    foutptr.close()

    shutil.copy("enzo.exe","%s/enzo.exe"%name)
    return name

def ploot(directory):
    plot_dir="%s/plots"%os.environ['HOME']

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

    fig,axes=plt.subplots(1,3, figsize=(12,4))

    for nds, ds in enumerate(ds_list):
        for nf, field in enumerate(fields):
            ray = ds.ortho_ray(0,[0.0,0.0])
            axes[nf].plot(ray['x'],ray[field])
            axes[nf].set(xlabel='x',ylabel='field')
    fig.tight_layout()
    fname = directory.split("/")[1]
    fig.savefig('%s/%s'%(plot_dir,fname))
    plt.close(fig)

def ploot_all(directory):
    plot_dir="%s/plots"%os.environ['HOME']

    OutputLogName = "%s/OutputLog"%directory
    fptr = open(OutputLogName,'r')
    lines = fptr.readlines()
    pfs = [line.split()[2] for line in lines]
    fptr.close()

    ds_list = [yt.load("%s/%s"%(directory, pf)) for pf in pfs]

    fields = ['density','pressure','velocity_x']

    fig,axes=plt.subplots(1,3, figsize=(12,4))

    for nds, ds in enumerate(ds_list):
        print(ds)
        for nf, field in enumerate(fields):
            ray = ds.ortho_ray(0,[0.0,0.0])
            axes[nf].plot(ray['x'],ray[field])
            axes[nf].set(xlabel='x',ylabel=field)
    fig.tight_layout()
    fig.savefig('%s/%s'%(plot_dir,directory))

