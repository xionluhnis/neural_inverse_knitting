import numpy as np


class Parameters:
    pass


hparams = Parameters()
hparams.model_type = 'NeutralNet'
hparams.checkpoint_dir = 'checkpoint'
hparams.is_train = True
hparams.dataset = 'dataset'
hparams.randseed = 2018
hparams.batch_size = 16  # 50
hparams.max_iter = 300000
hparams.learning_rate = 0.0001

arch_para = Parameters()
arch_para.resblk_DimCh = 64
arch_para.syn2real_DimCh = 64
arch_para.syn2real_Nresblk = 3
arch_para.mean_img_values = 0.5  # could be 1x1x1x3

arch_para.global_warp_DimCh = 64

arch_para.featext_for_warp_DimCh = 64
arch_para.featext_for_warp_Nresblk = 2
arch_para.featext_for_warp_DimFeat = 1000

arch_para.programsyn_DimCh = 64
arch_para.keypoint_decoder_DimCh = 256
arch_para.keypoint_decoder_no_layer_fc = 2

arch_para.texture_decoder_Dimch = 256
arch_para.texture_decoder_initial_downsample_rate = 16

arch_para.warp_mode = 1  # 1: , 2: TPS
arch_para.warp_reg = 0.001
arch_para.warp_no_boundary_pts = 2
