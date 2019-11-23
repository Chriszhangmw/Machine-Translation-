
import logging
import os

from tqdm import tqdm

from beam_search import BeamSearch
from data_load import get_batch
from hparams import Hparams
from model import Transformer
from utils import save_variable_specs,  import_tf

logging.basicConfig(level=logging.INFO)



logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
args = parser.parse_args()
hp = vars(args)
print(hp)
# import tensorflow
gpu_list = [str(i) for i in list(range(hp['gpu_nums']))]
tf = import_tf(gpu_list)
print(gpu_list)

train_batches, num_train_batches, num_train_samples = get_batch(hp['train_seg_x_dir'],
                                                                hp['train_seg_y_dir'],
                                                                hp,
                                                                hp['batch_size'],
                                                                hp['gpu_nums'],
                                                                shuffle=True)
handle = tf.placeholder(tf.string, shape=[])
iter = tf.data.Iterator.from_string_handle(handle, train_batches.output_types, train_batches.output_shapes)
xs, fre, ys = iter.get_next()
# create a iter of the correct shape and type
training_iter = train_batches.make_one_shot_iterator()


m = Transformer(hp)

# get op

loss, train_op, global_step, train_summaries,lr = m.train(xs,fre, ys)

from data_load import Vocab
vocab = Vocab(hp['vocab_path'],hp['vocab_size'])
token2idx = vocab.word2id
idx2token = vocab.id2word

bs = BeamSearch(m, hp['beam_size'], list(idx2token.keys())[2], list(idx2token.keys())[3], idx2token, hp['max_dec_len'], m.x,
                m.decoder_inputs, m.logits)
saver = tf.train.Saver(max_to_keep=hp['num_epochs'])
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    ckpt = tf.train.latest_checkpoint('./logdir')
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join('./logdir', "specs"))
    else:
        saver.restore(sess, ckpt)
    summary_writer = tf.summary.FileWriter('./logdir', sess.graph)

    # Iterator.string_handle() get a tensor that can be got value to feed handle placeholder
    training_handle = sess.run(training_iter.string_handle())
    total_steps = hp['num_epochs'] * num_train_batches
    _gs = sess.run(global_step)
    for i in tqdm(range(_gs, total_steps+1)):
        loss_,_, _gs, _summary,lr_ = sess.run([loss,train_op, global_step, train_summaries,lr], feed_dict={handle: training_handle})
        summary_writer.add_summary(_summary, _gs)
        print('loss is :', loss_,'learning rate is ',lr_)
        if _gs % (500) == 0 and _gs != 0:
            print("steps {} is done".format(_gs))
            ckpt_name = os.path.join('./model', './model')
            saver.save(sess, ckpt_name, global_step=_gs)
    summary_writer.close()
