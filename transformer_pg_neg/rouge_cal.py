


from hparams import Hparams
from pred import  Prediction

hparams = Hparams()
parser = hparams.parser
hppp = parser.parse_args()
hp = vars(hppp)
preds = Prediction(hp)

from data_load import Vocab



with open(hp['test_seg_x_dir'],'r',encoding='utf-8') as f_x:
    test_x = f_x.readlines()
with open(hp['test_seg_y_dir'],'r',encoding='utf-8') as f_y:
    text_y = f_y.readlines()
test = zip(test_x,text_y)


result_file = open('./result_neg.csv','w',encoding='utf-8')
tmp = []
for x,_ in test:
    # x = '款君越 变速箱 程序 可以 升级 吗 还是 只有 出厂 时 一个 标定 可以 升级 的 厂家 会 不定期 发布 新 的 软件 标号 变速箱 有 问题 吗 低速 换挡 很 干涩 动力 中断 明显 才 变速箱 有 故障 码 吗 有 过 一次 挂 档 不 走 档灯 闪烁 但是 挂 档 可以 然后 停 了 一 晚上 第二天 启动 又 好 了 那 应该 是 变速箱 有 问题 升级 软件 不 一定 可以 解决 起步 档 升档 的 时候 非常 慢去 店 升级 变速箱 程序 要 收费 吗 一般 是 没 收费 变速箱 有 故障 码 吗 没有 就是 换挡 干涩 低速 换挡 慢 动力 中断 感 明显 建议 先 更换 变速箱 油 再 升级 试试 公里 就要 换 变速箱 油 了 四五年 了 才 开 了 年 零个 月 出现 挂档 不动 的话 最好 检查一下 变速箱 油 就 那 一次 先到 店 升级 软件 试试 嗯 中午 下班 去 看看 好 的 之前 电话 问过 一家 店 他 说款 君越 用 的 是 第二代 稳定性 很 好奇 不 需要 升级 升级 可以 减少 换档 顿挫 感 优化 换挡 逻辑 是 的'
    result = preds.predict(x)[0]
    print(result)
    result_file.write(str(result) + '\n')

# vocab = Vocab(hp['vocab_path'], hp['vocab_size'])
# token2idx = vocab.word2id
# idx2token = vocab.id2word
# token2idx = token2idx
# idx2token = idx2token
# print(idx2token[7])