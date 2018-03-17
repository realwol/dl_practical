#-*- coding: utf-8 -*-
import numpy as np
from layers import *
from dropout_layers import *


class FullyConnectedNet(object):
  """
  深层全连接神经网络，其中隐藏层使用ReLU作为激活函数，输出层使用softmax作为分类器.
  该网络结构应该为:{affine - relu- [dropout]}x(L - 1) - affine - softmax
  """

  def __init__(self, input_dim=3*32*32,hidden_dims=[100,100],  num_classes=10,
               dropout=0, reg=0.0, weight_scale=1e-2, seed=None):
    """
    初始化全连接网络.
    
    Inputs:
    - input_dim: 输入维度
    - hidden_dims:  隐藏层各层维度，如[100,100]
    - num_classes: 分类数量.
    - dropout: 如果dropout = 0 表示不使用dropout.
    - reg:正则化衰减因子.
    - weight_scale:权重范围，给予初始化权重的标准差.
    - seed: 使用seed产生相同的随机数.
    """
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.params = {}


    layers_dims = [input_dim] + hidden_dims + [num_classes]
    for i in xrange(self.num_layers):
        self.params['W'+str(i+1)]=weight_scale*np.random.randn(
            layers_dims[i],layers_dims[i+1])
        self.params['b' + str(i + 1)] = np.zeros(
            (1, layers_dims[i + 1]))
    
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    

  def loss(self, X, y=None):
    mode = 'test' if y is None else 'train'
    # 设置执行模式
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    scores = None

    ############################################################################
    #                  任务：执行全连接网络的前馈过程。                        #
    #               计算数据的分类得分，将结果保存在scores中。                 #
    #      当使用dropout时，你需要使用self.dropout_param进行dropout前馈。      #
    #           例如 if self.use_dropout: dropout传播   else：正常传播         #
    ############################################################################





	
    ############################################################################
    #                             结束编码                                     #
    ############################################################################


    if mode == 'test':
      return scores

    loss, grads = 0.0, {}

    ############################################################################
    #      任务：实现全连接网络的反向传播。                                    #
    #        将损失值储存在loss中，梯度值储存在grads字典中                     #
    #     注意网络需要设置两种模式：有dropout，无dropout                       #
    #       例如if self.use_dropout: dropout传播，else：正常传播               #
    ############################################################################

	
	
	
	
    ############################################################################
    #                             结束编码                                     #
    ############################################################################

    return loss, grads
