#-*- coding: utf-8 -*-
import numpy as np
from layers import *
from dropout_layers import *
from bn_layers import *

class FullyConnectedNet(object):
  """
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  """

  def __init__(self, input_dim=3*32*32,hidden_dims=[100],  num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, seed=None):
    """
    初始化全连接网络.  
    Inputs:
    - input_dim: 输入维度
    - hidden_dims: 隐藏层各层维度向量，如[100,100]
    - num_classes: 分类个数.
    - dropout: 如果dropout=0，表示不使用dropout.
    - use_batchnorm：布尔型，是否使用BN
    - reg:正则化衰减因子.
    - weight_scale:权重初始化范围，标准差.
    - seed: 使用seed产生相同的随机数。
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.params = {}

    ############################################################################
    #               任务：初始化网络参数                                       #
    #          权重参数初始化和前面章节类似                                    #
    #          针对每一层神经元都要初始化对应的gamma和beta                     #
    #          如:第一层使用gamma1，beta1，第二层gamma2,beta2,                 #
    #           gamma初始化为1，beta初始化为0                                  # 
    ############################################################################

	
	
	
	
	
    ############################################################################
    #                            结束编码                                      #
    ############################################################################


    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed   

    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    

  def loss(self, X, y=None):
    # 设置执行模式
    mode = 'test' if y is None else 'train'

    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    #                  任务：执行全连接网络的前馈过程。                        #
    #               计算数据的分类得分，将结果保存在scores中。                 #
    #      当使用dropout时，你需要使用self.dropout_param进行dropout前馈。      #
    #      当使用BN时，self.bn_params[0]传到第一层，self.bn_params[1]第二层    #
    ############################################################################








    
    ############################################################################
    #                             结束编码                                     #
    ############################################################################

    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    #        任务：实现全连接网络的反向传播。                                  #
    #        将损失值储存在loss中，梯度值储存在grads字典中                     #
    #        当使用dropout时，需要求解dropout梯度                              #
    #        当使用BN时，需要求解BN梯度                                        #
    ############################################################################

	
	
	
	
	
	
	
	
    ############################################################################
    #                             结束编码                                     #
    ############################################################################

    return loss, grads
