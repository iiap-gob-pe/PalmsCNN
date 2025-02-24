import numpy as np
import skimage
import skimage.io
import scipy.io as sio
import skimage.transform
import sys

np.random.seed(0)

#VGG_MEAN = [103.939, 116.779, 123.68]
VGG_MEAN = [107.687, 132.034 , 77.097] #fron lista filtrada que tienen al menos un clase
CLASS_TO_SS = {"mauritia":1, "otra":2, "laotra":3}


def read_mat(path):
    return np.load(path)


def write_mat(path, m):
    np.save(path, m)


def read_ids(path):
    return [line.rstrip('\n') for line in open(path)]


class Batch_Feeder:
    def __init__(self, dataset, indices, train, batchSize, padWidth=None, padHeight=None, flip=False, keepEmpty=True):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._dataset = dataset
        self._indices = indices
        self._train = train
        self._batchSize = batchSize
        self._padWidth = padWidth
        self._padHeight = padHeight
        self._flip = flip
        self._keepEmpty = keepEmpty

    def set_paths(self, idList=None, imageDir=None, gtDir=None, ssDir=None):
        self._paths = []

        if self._train:
            for id in idList:
                id =id.strip()
                self._paths.append([id, imageDir + '/' + id + '_feature.npz',
                                    gtDir + '/' + id + '_dirmap.npz',
                                    ssDir + '/' + id + '_response.npz',
                                    ssDir + '/' + id + '_weightmap.npz'])
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
        ssBatch = []
        ssMaskBatch = []
        weightBatch = []

        if self._train:
            while(len(idBatch) < self._batchSize):
                npz=np.load(self._paths[self._index_in_epoch][3])#segmetacion
                ss = (npz.f.arr_0).astype(float)
                ss[ss==-9999]=0
                
                ssBinary, ssMask = self.ssProcess(ss[:,:,0])
               
                if ss.sum() > 0 or self._keepEmpty:
                    idBatch.append(self._paths[self._index_in_epoch][0])

                    npz=np.load(self._paths[self._index_in_epoch][1])#tiled img 3 band
                    image = (npz.f.arr_0).astype(float)
                    image[image==-9999]=0
                    image=self.image_scaling(image)

                    npz=np.load(self._paths[self._index_in_epoch][2])#Mapa de direccion
                    gt = (npz.f.arr_0).astype(float)
                    npz=np.load(self._paths[self._index_in_epoch][4])#pesos por tiled
                    weight =  (npz.f.arr_0[:,:,0]).astype(float)

                    imageBatch.append(self.pad(image[:,:,:3]))
                    gtBatch.append(self.pad(gt))
                    weightBatch.append(self.pad(weight))
                    ssBatch.append(self.pad(ssBinary))
                    ssMaskBatch.append(self.pad(ssMask))
                else:
                    pass
                    # raw_input("skipping " + self._paths[self._index_in_epoch][0])
                self._index_in_epoch += 1
                if self._index_in_epoch == self._numData:
                    self._index_in_epoch = 0
                    self.shuffle()

            imageBatch = np.array(imageBatch)
            gtBatch = np.array(gtBatch)
            ssBatch = np.array(ssBatch)
            ssMaskBatch = np.array(ssMaskBatch)
            weightBatch = np.array(weightBatch)

            if self._flip and np.random.uniform() > 0.5:
                for i in range(len(imageBatch)):
                    for j in range(3):
                        imageBatch[i,:,:,j] = np.fliplr(imageBatch[i,:,:,j])

                    weightBatch[i] = np.fliplr(weightBatch[i])
                    ssBatch[i] = np.fliplr(ssBatch[i])
                    ssMaskBatch[i] = np.fliplr(ssMaskBatch[i])

                    for j in range(2):
                        gtBatch[i,:,:,j] = np.fliplr(gtBatch[i,:,:,j])

                    gtBatch[i,:,:,0] = -1 * gtBatch[i,:,:,0]
            return imageBatch, gtBatch, weightBatch, ssBatch, ssMaskBatch, idBatch
        else:
            for example in self._paths[self._index_in_epoch:min(self._index_in_epoch+self._batchSize, self._numData)]:
           
                npz=np.load(example[1])#tiled img 3 band
                image = (npz.f.arr_0).astype(float)
                image[image==-9999]=0
                image=self.image_scaling(image)
                imageBatch.append(image[:,:,:3])

                idBatch.append(example[0])

                npz=np.load(example[2])#segmetacion
                ss = (npz.f.arr_0).astype(float)
                ss[ss==-9999]=0
                
                ssBinary, ssMask = self.ssProcess(ss[:,:,0])
                
                ssBatch.append(self.pad(ssBinary))
                ssMaskBatch.append(self.pad(ssMask))
            imageBatch = np.array(imageBatch)
            ssBatch = np.array(ssBatch)
            ssMaskBatch = np.array(ssMaskBatch)

            self._index_in_epoch += self._batchSize
            return imageBatch, ssBatch, ssMaskBatch, idBatch

    def total_samples(self):
        return self._numData

    def image_scaling(self, rgb_in):
        if rgb_in.dtype == np.float32:
            print(np.unique(rgb_in))
            rgb_in = rgb_in*255
            print(np.unique(rgb_in))
        elif rgb_in.dtype == np.uint8:
            rgb_in = rgb_in.astype(np.float32)

        rgb_out = np.zeros(rgb_in.shape).astype(np.float32)
        rgb_out[:,:,0] = rgb_in[:,:,0] - VGG_MEAN[0]
        rgb_out[:,:,1] = rgb_in[:,:,1] - VGG_MEAN[1]
        rgb_out[:,:,2] = rgb_in[:,:,2] - VGG_MEAN[2]

        return rgb_out

    def pad(self, data):
        if self._padHeight and self._padWidth:
            if data.ndim == 3:
                npad = ((0,self._padHeight-data.shape[0]),(0,self._padWidth-data.shape[1]),(0,0))
            elif data.ndim == 2:
                npad = ((0, self._padHeight - data.shape[0]), (0, self._padWidth - data.shape[1]))
            padData = np.pad(data, npad, mode='constant', constant_values=0)

        else:
            padData = data

        return padData

    def ssProcess(self,ssImage):
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

