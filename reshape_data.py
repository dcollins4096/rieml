import tube_loader
from importlib import reload
import pdb
reload(tube_loader)

if 0:
    data = tube_loader.load_many()#check_file='tubes_take5.h5')
    tube_loader.write_one('tubes_take6.h5',data)
if 1:
    data = tube_loader.load_many_time()#check_file='tubes_take5.h5')
    tube_loader.write_one('tubes_take7.h5',data)

if 0:
    def checker(datum, parameter):
        #rho, p, v
        a,b,c=datum[0]
        vals = np.zeros(6)
        vals[0]=a[ 0]
        vals[1]=a[-1]
        vals[2]=b[ 0]
        vals[3]=b[-1]
        vals[4]=c[ 0]
        vals[5]=c[-1]

        return ((parameter-vals)**2).sum()
    print('check 0',checker(data[60],parameters[60]))

