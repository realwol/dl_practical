#-*- coding: utf-8 -*-
import numpy as np

from layers import *
from rnn_layers import *


class CaptioningRNN(object):
  """
  处理图片说明任务RNN网络
  注意：不使用正则化
  """
  
  def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
               hidden_dim=128, cell_type='rnn'):
    """
    初始化CaptioningRNN 
    Inputs:
    - word_to_idx: 单词字典，用于查询单词索引对应的词向量
    - input_dim: 输入图片数据维度
    - wordvec_dim: 词向量维度.
    - hidden_dim: RNN隐藏层维度.
    - cell_type: 细胞类型; 'rnn' 或 'lstm'.
    """
    if cell_type not in {'rnn', 'lstm'}:
      raise ValueError('Invalid cell_type "%s"' % cell_type)
    
    self.cell_type = cell_type
    self.word_to_idx = word_to_idx
    self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
    self.params = {}
    
    vocab_size = len(word_to_idx)

    self._null = word_to_idx['<NULL>']
    self._start = word_to_idx.get('<START>', None)
    self._end = word_to_idx.get('<END>', None)
    
    # 初始化词向量
    self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
    self.params['W_embed'] /= 100
    
    # 初始化 CNN -> 隐藏层参数，用于将图片特征提取到RNN中
    self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
    self.params['W_proj'] /= np.sqrt(input_dim)
    self.params['b_proj'] = np.zeros(hidden_dim)

    # 初始化RNN参数
    dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
    self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
    self.params['Wx'] /= np.sqrt(wordvec_dim)
    self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
    self.params['Wh'] /= np.sqrt(hidden_dim)
    self.params['b'] = np.zeros(dim_mul * hidden_dim)
    
    # 初始化输出层参数 
    self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
    self.params['W_vocab'] /= np.sqrt(hidden_dim)
    self.params['b_vocab'] = np.zeros(vocab_size)
      

  def loss(self, features, captions):
    """
    计算RNN或LSTM的损失值。
    Inputs:
    - features: 输入图片特征(N, D)。
    - captions: 图像文字说明(N, T)。 
      
    Returns 元组:
    - loss: 损失值。
    - grads:梯度。
    """
    #将文字切分为两段：captions_in除去最后一词用于RNN输入
    #captions_out除去第一个单词，用于RNN输出配对
    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]
    
    # 掩码 
    mask = (captions_out != self._null)

    # 图像仿射转换矩阵
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    
    # 词嵌入矩阵
    W_embed = self.params['W_embed']

    # RNN参数
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

    # 隐藏层输出转化矩阵
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    
    loss, grads = 0.0, {}
    ############################################################################
    #            任务：实现CaptioningRNN传播                                   #
    #     (1)使用仿射变换(features,W_proj,b_proj)，                            #
    #           将图片特征输入进隐藏层初始状态h0(N,H)                          #
    #     (2)使用词嵌入层将captions_in中的单词索引转换为词向量(N,T,W)          #
    #     (3)使用RNN或LSTM处理词向量(N,T,H)                                    #
    #     (4)使用时序仿射传播temporal_affine_forward计算各单词得分(N,T,V)      #
    #     (5)使用temporal_softmax_loss计算损失值                               #
    ############################################################################

	
	
	
	
	
    ############################################################################
    #                             结束编码                                     #
    ############################################################################
    
    return loss, grads


  def sample(self, features, max_length=30):
    """
    测试阶段的前向传播过程，采样一批图片说明作为输入
    Inputs:
    - features: 图片特征(N, D).
    - max_length:生成说明文字的最大长度

    Returns:
    - captions: 说明文字的字典索引串(N, max_length)
    """
    N = features.shape[0]
    captions = self._null * np.ones((N, max_length), dtype=np.int32)

    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    W_embed = self.params['W_embed']
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    
    ###########################################################################
    #               任务：测试阶段前向传播                                    #
    #  提示:(1)第一个单词应该是<START>标记，captions[:,0]=self._start         #
    #       (2)当前单词输入为之前RNN的输出                                    #
    #    (3)前向传播过程为预测当前单词的下一个单词，                          #
    #     你需要计算所有单词得分，然后选取最大得分作为预测单词                #
    #    (4)你无法使用rnn_forward 或stm_forward函数，                         #
    #    你需要循环调用rnn_step_forward或lstm_step_forward函数                #
    ###########################################################################

	
	
	
	
	
	
    ############################################################################
    #                             结束编码                                     #
    ############################################################################
    return captions
