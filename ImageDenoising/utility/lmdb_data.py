"""Create lmdb dataset"""
from util import *
import lmdb
import scipy.io as scio

def create_lmdb_train(
    datadir, fns, name, matkey,
    crop_sizes, scales, ksizes, strides,
    load=h5py.File, augment=True,
    seed=2017):
    """
    Create Augmented Dataset
    """
    def preprocess(data):
        print('-------------------')
        print("Preprocessing data...")
        new_data = []
        data = minmax_normalize(data)
        print(f"Preprocess data shape: {data.shape}")
        # data = np.rot90(data, k=2, axes=(1,2)) # ICVL
        #data = minmax_normalize(data.transpose((2,0,1))) # for Remote Sensing
        # Visualize3D(data)
        if crop_sizes is not None:
            print('crop center to (%d, %d)' %(crop_sizes[0], crop_sizes[1]))
            data = crop_center(data, crop_sizes[0], crop_sizes[1])        
        
        for i in range(len(scales)):
            print('zooming to %f' %(scales[i]))
            if scales[i] != 1:
                temp = zoom(data, zoom=(1, scales[i], scales[i]))
            else:
                temp = data
            temp = Data2Volume(temp, ksizes=ksizes, strides=list(strides[i]))
            new_data.append(temp)
        for i,data in enumerate(new_data):
            print(f"new data {i} shape  {data.shape}")
        print('concatenating data...')
        new_data = np.concatenate(new_data, axis=0)
        print(f'new data shape after conc: {new_data.shape}')
        if augment:
            print('augmenting data...')
            for i in range(new_data.shape[0]):
                 new_data[i,...] = data_augmentation(new_data[i, ...])
        print(f"Postprocess data shape: {new_data.shape}")        
        return new_data.astype(np.float32)

    # Calculate map_size
    total_size = 0
    for fn in fns:
        try:
            X = load(datadir + fn)[matkey]
        except:
            print('loading', datadir+fn, 'fail')
            continue
        total_size += X.nbytes
    print(f"Map size(GB): {total_size / (1024 ** 3)}")
 
    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)        
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    print('loading mat: %s' %(fns[0]))
    data = load(datadir + fns[0])[matkey]
    data = preprocess(data)
    N = data.shape[0]
  
    path = os.path.abspath(name+'.db')
    print(path)
    manual_map = 10 * 1024 ** 3  # GB
    env = lmdb.open(path=path, map_size=manual_map, writemap=True)
    txt_file = open(os.path.join(name+'.db', 'meta_info.txt'), 'w')
    
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(fns):
            print('load mat: %s' %(fn))
            try:
                X = load(datadir + fn)[matkey]
            except:
                print('loading', datadir+fn, 'fail')
                continue
            X = preprocess(X)        
            N = X.shape[0]
            for j in range(N):
                c,h,w = X.shape[1:]
                data_byte = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txt_file.write(f'{str_id} ({h},{w},{c})\n')
                txn.put(str_id.encode('ascii'), data_byte)
    print('done')
        
def createDCmall():
    print('create wdc...')
    datadir = '/home/lofty/CODE/HyperSIGMA-fork/ImageDenoising/data/HSI_Data/Hyperspectral_Project/WDC/train/'
    fns = os.listdir(datadir) 
    
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    create_lmdb_train(
        datadir, fns, '/home/lofty/CODE/HyperSIGMA-fork/ImageDenoising/data/HSI_Data/Hyperspectral_Project/WDC/wdc', 'data',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(191, 64, 64),
        strides=[(191, 16, 16), (191, 8, 8), (191, 8, 8)],          
        load=scio.loadmat, augment=True,
    )

def createHYPSO_resized():
    print('create HYPSO resized...')
    datadir = '/home/lofty/CODE/HyperSIGMA-fork/ImageDenoising/data/HSI_Data/Hyperspectral_Project/HYPSO/train/'
    # datadir = './data/HSI_Data/Hyperspectral_Project/HYPSO/train/'
    fns = os.listdir(datadir) 
    
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    create_lmdb_train(
        datadir=datadir, 
        fns=fns, 
        name='/home/lofty/CODE/HyperSIGMA-fork/ImageDenoising/data/HSI_Data/Hyperspectral_Project/HYPSO/hypso', 
        # name='./data/HSI_Data/Hyperspectral_Project/HYPSO/hypso', 
        matkey='data',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(191, 64, 64),
        strides=[(191, 16, 16), (191, 8, 8), (191, 8, 8)],          
        load=scio.loadmat,
        augment=True
    )

def createHYPSO():
    print('create HYPSO...')
    datadir = '/home/lofty/CODE/HyperSIGMA-fork/ImageDenoising/data/HSI_Data/Hyperspectral_Project/HYPSO2/train/'
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    bands=120
    create_lmdb_train(
        datadir=datadir, 
        fns=fns, 
        name='/home/lofty/CODE/HyperSIGMA-fork/ImageDenoising/data/HSI_Data/Hyperspectral_Project/HYPSO2/hypso', 
        matkey='data',
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(bands, 64, 64),
        strides=[(bands, 16, 16), (bands, 8, 8), (bands, 8, 8)],          
        load=scio.loadmat,
        augment=True
    )

if __name__ == '__main__':
    # createDCmall()
    # createHYPSO_resized()
    createHYPSO()
