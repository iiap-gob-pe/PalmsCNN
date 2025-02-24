import numpy as np
import skimage
import skimage.io
import scipy.io as sio
import scipy.misc
import skimage.transform

np.random.seed(0)

#VGG_MEAN = [103.939, 116.779, 123.68]
#VGG_MEAN = [107.687 132.034  77.097] #fron lista filtrada que tienen al menos un clase
CLASS_TO_SS = {"mauritia":1, "otra":2, "laotra":3}

def read_mat(path):
    return np.load(path)


def write_mat(path, m):
    np.save(path, m)

class Batch_Feeder:
    def __init__(self, dataset, train, batchSize, padWidth=None, padHeight=None, flip=False, keepEmpty=True, shuffle=False):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._dataset = dataset
        self._train = train
        self._batchSize = batchSize
        self._padWidth = padWidth
        self._padHeight = padHeight
        self._flip = flip
        self._keepEmpty = keepEmpty
        self._shuffle = shuffle

    def set_paths(self, idList=None, imageDir=None, gtDir=None, ssDir=None):
        self._paths = []

        if self._train:
            for id in idList:
                id =id.strip()
                self._paths.append([id, 
                                    imageDir + '/' + id + '_feature.npz',
                                    gtDir + '/' + id + '_deepmap.npz',
                                    ssDir + '/' + id + '_response.npz',
                                    ssDir + '/' + id + '_weightmap.npz'])
            if self._shuffle:
                self.shuffle()
        else:
            for id in idList:
                id =id.strip()
                self._paths.append([id, imageDir + '/' + id + '_feature.npz',
                                    ssDir + '/' + id + '_response.npz'])

        self._numData = len(self._paths)

        if self._numData < self._batchSize:
            self._batchSize = self._numData

    def shuffle(self):
        np.random.shuffle(self._paths)

    def next_batch(self):
        idBatch = []
        imageBatch = []
        gtBatch = []
        ssBinaryBatch = []
        ssMaskBatch = []
        weightBatch = []

        if self._train:
            while(len(idBatch) < self._batchSize):
                #ssImage = skimage.io.imread(self._paths[self._index_in_epoch][3])
                npz=np.load(self._paths[self._index_in_epoch][3])#segmetacion
                ssImage = (npz.f.arr_0).astype(float)
                ssImage[ssImage==-9999]=0
                ssBinary, ssMask = ssProcess(ssImage[:,:,0])
    

                idBatch.append(self._paths[self._index_in_epoch][0])#nombre tiled
                #image = (image_scaling(skimage.io.imread(self._paths[self._index_in_epoch][1]))).astype(float)
                npz=np.load(self._paths[self._index_in_epoch][1])#tiled img 3 band
                image = (npz.f.arr_0).astype(float)
                image[image==-9999]=0
                #image = scipy.misc.imresize(image, 50)
                #gt = (sio.loadmat(self._paths[self._index_in_epoch][2])['depth_map']).astype(float)
                npz=np.load(self._paths[self._index_in_epoch][2])#Mapa de profundidad
                gt = (npz.f.arr_0).astype(float)
                #weight = (sio.loadmat(self._paths[self._index_in_epoch][2])['weight_map']).astype(float)
                npz=np.load(self._paths[self._index_in_epoch][4])#pesos por tiled
                weight =  (npz.f.arr_0[:,:,0]).astype(float)

                imageBatch.append(pad(image[:,:,:3], self._padHeight, self._padWidth))
                gtBatch.append(pad(gt, self._padHeight, self._padWidth))
                weightBatch.append(pad(weight, self._padHeight, self._padWidth))
                ssBinaryBatch.append(pad(ssBinary, self._padHeight, self._padWidth))
                ssMaskBatch.append(pad(ssMask, self._padHeight, self._padWidth))

                self._index_in_epoch += 1

                if self._index_in_epoch == self._numData:
                    self._index_in_epoch = 0
                    if self._shuffle:
                        self.shuffle()

            imageBatch = np.array(imageBatch)
            gtBatch = np.array(gtBatch)
            ssBinaryBatch = np.array(ssBinaryBatch)
            ssMaskBatch = np.array(ssMaskBatch)
            weightBatch = np.array(weightBatch)

            if self._flip and np.random.uniform() > 0.5:
                for i in range(len(imageBatch)):
                    for j in range(1):
                        imageBatch[i,:,:,j] = np.fliplr(imageBatch[i,:,:,j])

                    ssBinaryBatch[i] = np.fliplr(ssBinaryBatch[i])
                    ssMaskBatch[i] = np.fliplr(ssMaskBatch[i])
                    gtBatch[i] = np.fliplr(gtBatch[i])
                    weightBatch[i] = np.fliplr(weightBatch[i])

            return imageBatch, gtBatch, weightBatch, ssBinaryBatch, ssMaskBatch, idBatch
        else:
            for example in self._paths[self._index_in_epoch:min(self._index_in_epoch+self._batchSize, self._numData)]:
                #image = skimage.io.imread(example[1])
                #image = scipy.misc.imresize(image,50)
                npz=np.load(example[1])
                image = (npz.f.arr_0).astype(float)
                #image = pad(image_scaling(image), self._padHeight, self._padWidth).astype(float)
                image = pad(image[:,:,:3], self._padHeight, self._padWidth).astype(float)

                imageBatch.append(image)

                idBatch.append(example[0])
                #ssImage = skimage.io.imread(example[2])
                npz=np.load(example[2])
                ssImage = (npz.f.arr_0).astype(float)

                #ssImage = scipy.misc.imresize(ssImage, 50, interp="nearest")

                ssBinary, ssMask = ssProcess(ssImage[:,:,0])

                ssMaskBatch.append(pad(ssMask, self._padHeight, self._padWidth))
                ssBinaryBatch.append(pad(ssBinary, self._padHeight, self._padWidth))

            imageBatch = np.array(imageBatch)
            ssBinaryBatch = np.array(ssBinaryBatch)
            ssMaskBatch = np.array(ssMaskBatch)

            self._index_in_epoch += self._batchSize

            return imageBatch, ssBinaryBatch, ssMaskBatch, idBatch

    def total_samples(self):
        return self._numData

def read_ids(path):
    # return ['munster/munster_000071_000019']
    return [line.rstrip('\n') for line in open(path)]

def image_scaling(rgb_in):
    if rgb_in.dtype == np.float32:
        rgb_in = rgb_in*255
    elif rgb_in.dtype == np.uint8:
        rgb_in = rgb_in.astype(np.float32)

    # VGG16 was trained using opencv which reads images as BGR, but skimage reads images as RGB
    rgb_out = np.zeros(rgb_in.shape).astype(np.float32)
    rgb_out[:,:,0] = rgb_in[:,:,2] - VGG_MEAN[2]
    rgb_out[:,:,1] = rgb_in[:,:,1] - VGG_MEAN[1]
    rgb_out[:,:,2] = rgb_in[:,:,0] - VGG_MEAN[0]

    return rgb_out

def pad(data, padHeight=None, padWidth=None):
    if padHeight and padWidth:
        if data.ndim == 3:
            npad = ((0,padHeight-data.shape[0]),(0,padWidth-data.shape[1]),(0,0))
        elif data.ndim == 2:
            npad = ((0, padHeight - data.shape[0]), (0, padWidth - data.shape[1]))
        padData = np.pad(data, npad, mode='constant', constant_values=0)

    else:
        padData = data

    return padData

def ssProcess(ssImage):
    ssMask = np.zeros(shape=ssImage.shape, dtype=np.float32)
    ssImageInt = ssImage
    #ssMask = ssImage
    if ssImageInt.dtype == np.float32:
        ssImageInt = (ssImageInt*255).astype(np.uint8)

    # order: Person, Rider, Motorcycle, Bicycle, Car, Truck, Bus, Train

    ssMask += (ssImageInt==CLASS_TO_SS['mauritia']).astype(np.float32)*1
    ssMask += (ssImageInt==CLASS_TO_SS['otra']).astype(np.float32)*2
    ssMask += (ssImageInt==CLASS_TO_SS['laotra']).astype(np.float32)*3


    ssBinary = (ssMask != 0).astype(np.float32)

    #ssMask[ssMask == 0] = 1 # temp fix

    ssMask = (ssMask - 5) * 32

    return ssBinary, ssMask



