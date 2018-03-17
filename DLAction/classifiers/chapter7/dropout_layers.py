#-*- coding: utf-8 -*-
import numpy as np
from layers import *
def dropout_forward(x, dropout_param):
  """
  执行dropout前向传播
  Inputs:
  - x: 输入数据
  - dropout_param: 字典类型，使用下列键值:
    - p: dropout参数。每个神经元的激活概率p
    - mode: 'test'或'train'. 训练模式使用dropout;测试模式仅仅返回输入值。
    - seed: 随机数生成种子. 

  Outputs:
  - out: 和输入数据相同形状
  - cache:元组(dropout_param, mask). 
          训练模式时，掩码mask用于激活该层神经元，测试模式时不使用
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    #                任务：执行dropout训练模式前向传播。                      #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < p)/p
    out =x*mask
    ###########################################################################
    #                           结束编码                                      #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    #               任务： 执行测试阶段dropout前向传播。                      #
    ###########################################################################
    out = x
    ###########################################################################
    #                           结束编码                                      #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  dropout反向传播

  Inputs:
  - dout: 上层梯度
  - cache: dropout_forward中的缓存(dropout_param, mask)。
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    #                      任务：实现dropout反向传播                          #
    ###########################################################################
    dx =dout*mask
    ###########################################################################
    #                            结束编码                                     #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx

def affine_relu_dropout_forward(x,w,b,dropout_param):
  """
  组合affine_relu_dropout前向传播
  Inputs:
  - x: 输入数据
  - w: 权重参数
  - b: 偏置项
  - dropout_param: 字典类型，使用下列键值:
    - p: dropout参数。每个神经元的激活概率p
    - mode: 'test'或'train'. 训练模式使用dropout;测试模式仅仅返回输入值。
    - seed: 随机数生成种子. 

  Outputs:
  - out: 和输入数据相同形状
  - cache:缓存包含(cache_affine,cache_relu,cache_dropout)
  """ 
  out_dropout = None
  cache =None
  #############################################################################
  #               任务: 实现 affine_relu_dropout 神经元前向传播.              #
  #         注意：你需要调用affine_forward以及relu_forward函数，              #
  #              并将各自的缓存保存在cache中                                  #
  #############################################################################  
  out_affine, cache_affine = affine_forward(x,w,b)
  out_relu,cache_relu =relu_forward(out_affine)
  out_dropout,cache_dropout =dropout_forward(out_relu,dropout_param)
  cache = (cache_affine,cache_relu,cache_dropout)
  ###########################################################################
  #                            结束编码                                     #
  ###########################################################################    
  return out_dropout,cache

def affine_relu_dropout_backward(dout,cache):
  """
   affine_relu_dropout神经元的反向传播
   
  Input:
  - dout: 上层误差梯度
  - cache: 缓存(cache_affine,cache_relu,cache_dropout)

  Returns:
  - dx: 输入数据x的梯度
  - dw: 权重矩阵w的梯度
  - db: 偏置向量b的梯度
  """  
  cache_affine,cache_relu,cache_dropout = cache
  dx,dw,db=None,None,None
  ###########################################################################
  #               任务：实现affine_relu_dropout反向传播                     #
  ###########################################################################  
  ddropout = dropout_backward(dout,cache_dropout)
  drelu = relu_backward(ddropout,cache_relu)
  dx,dw,db = affine_backward(drelu,cache_affine)
  ###########################################################################
  #                          结束编码                                      #
  ###########################################################################
  return dx,dw,db




