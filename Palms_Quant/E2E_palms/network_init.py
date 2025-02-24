import sys
sys.path.append('../code/Palms_Quant/E2E_palms')
import e2e_model 
def get_model(wd=None, modelWeightPaths=None):
    params = {
        "direction/conv1_1": {"name": "direction/conv1_1", "shape": [3, 3, 4, 64], "std": None, "act": "relu"}, # 1 capa que extre caracteristicas
        "direction/conv1_2": {"name": "direction/conv1_2", "shape": [3, 3, 64, 64], "std": None, "act": "relu"}, # 2 capa que extre caracteristicas
        "direction/conv2_1": {"name": "direction/conv2_1", "shape": [3, 3, 64, 128], "std": None, "act": "relu"},
        "direction/conv2_2": {"name": "direction/conv2_2", "shape": [3, 3, 128, 128], "std": None, "act": "relu"},
        "direction/conv3_1": {"name": "direction/conv3_1", "shape": [3, 3, 128, 256], "std": None, "act": "relu"},
        "direction/conv3_2": {"name": "direction/conv3_2", "shape": [3, 3, 256, 256], "std": None, "act": "relu"},
        "direction/conv3_3": {"name": "direction/conv3_3", "shape": [3, 3, 256, 256], "std": None, "act": "relu"},
        "direction/conv4_1": {"name": "direction/conv4_1", "shape": [3, 3, 256, 512], "std": None, "act": "relu"},
        "direction/conv4_2": {"name": "direction/conv4_2", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "direction/conv4_3": {"name": "direction/conv4_3", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "direction/conv5_1": {"name": "direction/conv5_1", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "direction/conv5_2": {"name": "direction/conv5_2", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "direction/conv5_3": {"name": "direction/conv5_3", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "direction/fcn5_1": {"name": "direction/fcn5_1", "shape": [5, 5, 512, 512], "std": None, "act": "relu"},
        "direction/fcn5_2": {"name": "direction/fcn5_2", "shape": [1, 1, 512, 512], "std": None, "act": "relu"},
        "direction/fcn5_3": {"name": "direction/fcn5_3", "shape": [1, 1, 512, 256], "std": None, "act": "relu"},
        "direction/upscore5_3": {"name": "direction/upscore5_3", "ksize": 8, "stride": 4, "outputChannels": 256},
        "direction/fcn4_1": {"name": "direction/fcn4_1", "shape": [5, 5, 512, 512], "std": None, "act": "relu"},
        "direction/fcn4_2": {"name": "direction/fcn4_2", "shape": [1, 1, 512, 512], "std": None, "act": "relu"},
        "direction/fcn4_3": {"name": "direction/fcn4_3", "shape": [1, 1, 512, 256], "std": None, "act": "relu"},
        "direction/upscore4_3": {"name": "direction/upscore4_3", "ksize": 4, "stride": 2, "outputChannels": 256},
        "direction/fcn3_1": {"name": "direction/fcn3_1", "shape": [5, 5, 256, 256], "std": None, "act": "relu"},
        "direction/fcn3_2": {"name": "direction/fcn3_2", "shape": [1, 1, 256, 256], "std": None, "act": "relu"},
        "direction/fcn3_3": {"name": "direction/fcn3_3", "shape": [1, 1, 256, 256], "std": None, "act": "relu"},
        "direction/fuse3_1": {"name": "direction/fuse_1", "shape": [1,1,256*3,512], "std": None, "act": "relu"},
        "direction/fuse3_2": {"name": "direction/fuse_2", "shape": [1, 1, 512, 512], "std": None, "act": "relu"},
        "direction/fuse3_3": {"name": "direction/fuse_3", "shape": [1, 1, 512, 2], "std": None, "act": "lin"},
        "direction/upscore3_1": {"name": "direction/upscore3_1", "ksize": 8, "stride": 4, "outputChannels": 2},

        "depth/conv1_1": {"name": "depth/conv1_1", "shape": [5,5,2,64], "std": None, "act": "relu"},
        "depth/conv1_2": {"name": "depth/conv1_2", "shape": [5,5,64,128], "std": None, "act": "relu"},
        "depth/conv2_1": {"name": "depth/conv2_1", "shape": [5,5,128,128], "std": None, "act": "relu"},
        "depth/conv2_2": {"name": "depth/conv2_2", "shape": [5,5,128,128], "std": None, "act": "relu"},
        "depth/conv2_3": {"name": "depth/conv2_3", "shape": [5,5,128,128], "std": None, "act": "relu"},
        "depth/conv2_4": {"name": "depth/conv2_4", "shape": [5,5,128,128], "std": None, "act": "relu"},
        "depth/fcn1": {"name": "depth/fcn1", "shape": [1,1,128,128], "std": None, "act": "relu"},
        "depth/fcn2": {"name": "depth/fcn2", "shape": [1,1,128,16], "std": None, "act": "relu"},
        "depth/upscore": {"name": "depth/upscore", "ksize": 8, "stride": 4, "outputChannels": 16},
        }
    
    return e2e_model.Network(params, wd=wd, modelWeightPaths=modelWeightPaths)
