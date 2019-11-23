# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
date: 2019/5/21
mail: cally.maxiong@gmail.com
page: http://www.cnblogs.com/callyblog/
'''
import os
import re
from beam_search import BeamSearch
from data_load import Vocab
from hparams import Hparams
from model import Transformer




import pickle
import numpy as np
def load_obj():
    with open('./data/frequence_neg.pkl', 'rb') as f:
        return pickle.load(f)
fre = load_obj()
a = min(fre.values())
word_freqs = np.array(list(fre.values())).sum()

def import_tf(device_id=-1, verbose=False):
    """
    import tensorflow, set tensorflow graph load device, set tensorflow log level, return tensorflow instance
    :param device_id: GPU id
    :param verbose: tensorflow logging level
    :return: tensorflow instance
    """
    # set visible gpu, -1 is cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf

class Prediction:
    def __init__(self, hp):
        """
        :param model_dir: model dir path
        :param vocab_file: vocab file path
        """
        self.tf = import_tf(0)

        self.hp = hp
        self.model_dir = './model'
        vocab = Vocab(hp['vocab_path'], hp['vocab_size'])
        token2idx = vocab.word2id
        idx2token = vocab.id2word
        self.token2idx = token2idx
        self.idx2token = idx2token

        self.model = Transformer(self.hp)

        self._add_placeholder()
        self._init_graph()

    def _init_graph(self):
        """
        init graph
        """
        self.ys = (self.input_y, None, None)
        self.xs = (self.input_x, None)
        self.f = (self.fre, None)
        self.memory = self.model.encode(self.xs,False)[0]
        self.logits = self.model.decode(self.xs,self.f,self.ys,self.memory, False)[0]

        ckpt = self.tf.train.get_checkpoint_state(self.model_dir).all_model_checkpoint_paths[-1]

        graph = self.logits.graph
        sess_config = self.tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        saver = self.tf.train.Saver()
        self.sess = self.tf.Session(config=sess_config, graph=graph)

        self.sess.run(self.tf.global_variables_initializer())
        self.tf.reset_default_graph()
        saver.restore(self.sess, ckpt)

        self.bs = BeamSearch(self.model,
                             self.hp['beam_size'],
                             list(self.idx2token.keys())[2],
                             list(self.idx2token.keys())[3],
                             self.idx2token,
                             self.hp['max_dec_len'],
                             self.input_x,
                             self.input_y,
                             self.logits)

    def predict(self, content):
        """
        abstract prediction by beam search
        :param content: article content
        :return: prediction result
        """
        input_x = list(content)
        while len(input_x) < self.hp['max_enc_len']: input_x.append('<pad>')
        input_x = input_x[:self.hp['max_enc_len']]
        # dis = [fre.get(w, 0) / word_freqs for w in input_x]
        dis = [(np.sqrt(fre.get(w, a) / a + 1) * a / fre.get(w, a)) / 2 for w in input_x]
        # dis = [1 for _ in input_x]

        input_x = [self.token2idx.get(s, self.token2idx['UNK']) for s in input_x]
        # print('$$$'*20)
        # print(input_x)
        # # print(dis)
        # print('$$$' * 20)
        print(input_x)
        memory = self.sess.run(self.memory, feed_dict={self.input_x: [input_x]})
        print('memory is ',memory)

        return self.bs.search(self.sess, input_x, memory[0])

    def _add_placeholder(self):
        """
        add tensorflow placeholder
        """
        self.input_x = self.tf.placeholder(dtype=self.tf.int32, shape=[None, self.hp['max_enc_len']], name='input_x')
        self.fre = self.tf.placeholder(dtype=self.tf.float32, shape=[None, self.hp['max_enc_len']])
        self.input_y = self.tf.placeholder(dtype=self.tf.int32, shape=[None, None], name='input_y')




if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hppp = parser.parse_args()
    hp = vars(hppp)
    preds = Prediction(hp)
    content = '我 的 帕萨特 烧 机油 怎么办 怎么办 请问 你 的 车 跑 了 多少 公里 了 如果 在 保修期 内 可以 到 当地 的 店 里面 进行 检查 维修 如果 已经 超出 了 保修期 建议 你 到 当地 的 大型 维修 店 进行 检查 烧 机油 一般 是 发动机 活塞环 间隙 过大 和 气门 油封 老化 引起 的 如果 每 公里 烧一升 机油 的话 可以 在 后备箱 备 一些 机油 以便 机油 报警 时有 机油 及时 补充 如果 超过 两升 或者 两升 以上 建议 你 进行 发动机 检查 维修 嗯'
    # content = '我 的 是 新款 君威 今晚 开车 的 时候 突然 有 一股 从 发动机舱 出来 的 烧焦 味 是 什么 原因 呢 师傅 车子 有 什么 故障 或者 异常 吗 没什么 故障 也 没 异常 就是 突然 烧 了 什么 一样 一股 异味 师傅 车子 没有 异常 那 可能 排气管 关上 缠到 了 方 编袋 造成 的 或者 车子 开 到 修理厂 清理 一下 希望 能够 帮到 你 祝 你 生活 愉快'
    # content = '我 的 是 宝马 昨天 刚 在 店 做 完 保养 本来 挺 好 跑 了 一阵 发现 左 后 轮胎 面 就 扎 了 钉 因为 晚上 晚 了 随便 找 了 个 补胎 说 钉 扎 的 非常 小贴 了 双层 胶片 较 了 胎 压前 两个 轮较 到 了 后 两个 轮 然后 初始 了 胎压 检测 开 回来 正常 但 一上 就 感觉 不 稳 请问 这样 还有 必要 到 店 做 检测 吗 可以 去 做 一个 轮胎 动平衡'
    result = preds.predict(content)
    print('原文如下：',content)
    print('#'*30)
    print('摘要如下：')
    for res in result:
        res = str(res).replace('UNK','')
        if 'STOP' in res:
            res = res.split('STOP')[0]
        print(res)