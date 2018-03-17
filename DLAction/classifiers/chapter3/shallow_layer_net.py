#-*- coding: utf-8 -*-
import numpy as np
from layers import *

class ShallowLayerNet(object):
    """
    浅层全连接神经网络，其中隐藏层使用ReLU作为激活函数，输出层使用softmax作为分类器
    该网络结构应该为     affine - relu -affine - softmax
    """
    
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
        """
        初始化网络.

        Inputs:
        - input_dim: 输入数据维度
        - hidden_dim: 隐藏层维度
        - num_classes: 分类数量
        - weight_scale: 权重范围，给予初始化权重的标准差
        - reg: L2正则化的权重衰减系数.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        #                      任务：初始化权重以及偏置项                          #
        #      权重应该服从标准差为weight_scale的高斯分布，偏置项应该初始化为0,    #
        #        所有权重矩阵和偏置向量应该存放在self.params字典中。               #
        #     第一层的权重和偏置使用键值 'W1'以及'b1'，第二层使用'W2'以及'b2'      #
        ############################################################################

		
		
		
		
        ############################################################################
        #                           结束编码                                       #
        ############################################################################
    
    
    
    def loss(self, X, y=None):
        """
        计算数据X的损失值以及梯度.

        Inputs:
        - X: 输入数据，形状为(N, d_1, ..., d_k)的numpy数组。
        - y: 数据类标，形状为(N,)的numpy数组。

        Returns:
        如果y为 None, 表明网络处于测试阶段直接返回输出层的得分即可:
        - scores:形状为 (N, C)，其中scores[i, c] 是数据 X[i] 在第c类上的得分.
        
        如果y为 not None, 表明网络处于训练阶段，返回一个元组:
        - loss:数据的损失值
        - grads: 与参数字典相同的梯度字典，键值和参数字典的键值要相同
        """  
        scores = None
        ############################################################################
        #              任务: 实现浅层网络的前向传播过程，                          #
        #                       计算各数据的分类得分                               #
        ############################################################################

		
		

        ############################################################################
        #                            结束编码                                     #
        ############################################################################

        # 如果y为 None 直接返回得分
        if y is None:
          return scores

        loss, grads = 0, {}
        ############################################################################
        #                    任务：实现浅层网络的反向传播过程，                    #
        #            将损失值存储在loss中,将各层梯度储存在grads字典中。            #
        #                           注意：别忘了还要计算权重衰减哟。               #
        ############################################################################

		
		
		
        ############################################################################
        #                             结束编码                                     #
        ############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
        """
        使用随机梯度下贱训练神经网络
        Inputs:
        - X: 训练数据
        - y: 训练类标.
        - X_val: 验证数据.
        - y_val:验证类标.
        - learning_rate: 学习率.
        - learning_rate_decay: 学习率衰减系数
        - reg: 权重衰减系数.
        - num_iters: 迭代次数.
        - batch_size: 批量大小.
        - verbose:是否在训练过程中打印结果.
        """
        num_train = X.shape[0]
        self.reg =reg
        #打印以及学习率衰减周期
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        best_val=-1
        for it in xrange(num_iters):
          X_batch = None
          y_batch = None

          sample_index = np.random.choice(num_train, batch_size, replace=True)
          X_batch = X[sample_index, :]  # (batch_size,D)
          y_batch = y[sample_index]  # (1,batch_size)

          #计算损失及梯度
          loss, grads = self.loss(X_batch, y=y_batch)
          loss_history.append(loss)

          #修改权重
          self.params['W1'] += -learning_rate*grads['W1']
          self.params['W2'] += -learning_rate*grads['W2']
          self.params['b1'] += -learning_rate*grads['b1']
          self.params['b2'] += -learning_rate*grads['b2']

          if verbose and it % 100 == 0:
            print '迭代次数 %d / %d: 损失值 %f' % (it, num_iters, loss)

          if it % iterations_per_epoch == 0:
            # 测试精度
            train_acc = (self.predict(X_batch) == y_batch).mean()
            val_acc = (self.predict(X_val) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            if (best_val < val_acc):
                best_val = val_acc
            # 学习率衰减
            learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
          'best_val_acc':best_val
        }
    
    
    def predict(self, X):
        """
        Inputs:
        - X: 输入数据
        Returns:
        - y_pred: 预测类别
        """
        y_pred = None
        out1,cache1 = affine_relu_forward(X,self.params['W1'],self.params['b1'])
        scores,cache2 = affine_forward(out1,self.params['W2'],self.params['b2'])
        y_pred = np.argmax(scores, axis=1)
        return y_pred