import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import sys
import pandas as pd
import scipy
from yahoo_finance import Share


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def stock_relativized_state(OHLCV):
    return np.array([(OHLCV[-1,3] - array)/OHLCV[-1,3] for array in OHLCV.T[:-1]] + [np.float32(OHLCV.T[-1]).tolist()]).T


class StockTradeSimulator():
    def __init__(self, data, max_episode_len, window_size):
        self.data = np.array(data)
        self.max_episode_len = max_episode_len
        self.window_size = window_size
        self.trade_log = []

    def test(self):
        self.index = self.window_size
        self.trades = []
        self.principal = 1.
        self.invested = 0.
        self.trade_data = []

    def get_state(self):
        data = self.data[(self.index-self.window_size):self.index,:]
        return [data, self.trades]

    def is_terminal(self):
        if self.index == len(self.data)-1:
            return 1
        return 0

    def open_trades(self, longInit, longSize, shortInit, shortSize):
        if self.invested == 1.0:
            return 0
        if shortSize + longSize + self.invested > 1.0:
            scalar = (1 - self.invested) / (shortSize*shortInit + longSize*longInit)
            longSize *= scalar
            shortSize *= scalar
        if shortInit:
            self.trades.append([-1, shortSize])
            self.trade_data.append([self.index, self.principal * shortSize])
            self.invested += shortSize
            # self.num_trades += 1
        if longInit:
            self.trades.append([1, longSize])
            self.trade_data.append([self.index, self.principal * longSize])
            self.invested += longSize
            # self.num_trades += 1
        return 0

    def close_trade(self, id):
        self.invested -= self.trades[id][1]
        self.trade_log.append(self.trades[id]+[self.trade_data[id][0]]+[self.index])

        if self.trades[id][0] > 0: # long
            self.principal += self.trade_data[id][1] * (self.data[self.index,3] - self.data[self.trade_data[id][0],3]) / \
                                  self.data[self.trade_data[id][0],3]
            del self.trades[id]
            del self.trade_data[id]
            return 0
        else:
            self.principal += self.trade_data[id][1] * (self.data[self.trade_data[id][0],3] - self.data[self.index,3]) / \
                                  self.data[self.index, 3]
            del self.trades[id]
            del self.trade_data[id]
            return 0

    def step(self, actions):
        # # to avoid stagnation:
        # if len(self.trades)==0:
        #     self.inactive_steps += 1

        # actions take form: [[longInit, longSize, shortInit, shortSize], [trade1_close, ..., tradeN_close]]
        if len(actions[1])!=len(self.trades):
            raise ValueError('Length of actions[1] must equal number of open trades')

        r = self.open_trades(*actions[0])
        for a, action in reversed(list(enumerate(actions[1]))):
            if action > 0:
                r += self.close_trade(a)

        self.index += 1
        price_growth = (self.data[self.index,3] - self.data[self.index-1,3])/self.data[self.index-1,3]

        r += sum([price_growth*trade[0]*trade[1] for trade in self.trades])
        # r -= (2**(self.inactive_steps**4/300.**4) - 1 )/10

        return r, self.is_terminal(), self.get_state()


class AC_Network():
    def __init__(self, window_size, scope, trainer):
        with tf.variable_scope(scope):
            # Inputs
            self.price_input = tf.placeholder(shape=[None, window_size, 4], dtype=tf.float32, name='price_input')
            self.volume_input = tf.placeholder(shape=[None, window_size, 1], dtype=tf.float32, name='volume_input')
            self.trade_input = tf.placeholder(shape=[None, None, 2], dtype=tf.float32, name='trade_input')

            # Price Convolutions
            priceIn = tf.reshape(tf.pad(self.price_input, [[0, 0], [1, 1], [0, 0]]), shape=[-1, window_size + 2, 4, 1])
            priceConv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=priceIn, num_outputs=32,
                                     kernel_size=[3, 4], stride=[1, 1], padding='VALID')
            priceConv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=priceConv1, num_outputs=32,
                                     kernel_size=[6, 1], stride=[1, 1], padding='SAME')
            priceConv3 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=priceConv2, num_outputs=32,
                                     kernel_size=[6, 1], stride=[1, 1], padding='SAME')
            priceConv4 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=priceConv3, num_outputs=32,
                                     kernel_size=[6, 1], stride=[1, 1], padding='SAME')
            priceMod = tf.reshape(priceConv4, [-1, 1, 32 * window_size])

            # Volume Convolutions
            volumeIn = tf.reshape(tf.pad(self.volume_input, [[0, 0], [1, 1], [0, 0]]),
                                  shape=[-1, window_size + 2, 1, 1])
            # batchnorm?
            volumeConv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                      inputs=volumeIn, num_outputs=32,
                                      kernel_size=[3, 1], stride=[1, 1], padding='VALID')
            volumeConv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                      inputs=volumeConv1, num_outputs=32,
                                      kernel_size=[6, 1], stride=[1, 1], padding='SAME')
            volumeConv3 = slim.conv2d(activation_fn=tf.nn.elu,
                                      inputs=volumeConv2, num_outputs=32,
                                      kernel_size=[6, 1], stride=[1, 1], padding='SAME')
            volumeConv4 = slim.conv2d(activation_fn=tf.nn.elu,
                                      inputs=volumeConv3, num_outputs=32,
                                      kernel_size=[6, 1], stride=[1, 1], padding='SAME')
            volumeMod = tf.reshape(volumeConv4, [-1, 1, 32 * window_size])

            # Merge Convs
            conv_dims = 256
            mergedConvs = tf.concat((priceMod, volumeMod), axis=2, name='mergedConvs')
            convDense = slim.fully_connected(mergedConvs, conv_dims, activation_fn=tf.nn.elu)

            # State of actives trades
            trade_dims = 12
            batch_size = tf.shape(self.trade_input)[0]
            num_trades = tf.shape(self.trade_input)[1]
            tradeStates = tf.reshape(slim.fully_connected(self.trade_input, trade_dims, activation_fn=tf.nn.elu),
                                     (batch_size, num_trades, trade_dims))

            sumTrades = tf.reshape(tf.reduce_sum(tradeStates, 1), (batch_size, 1, trade_dims))
            tradeDense = tf.concat((tradeStates, tf.tile(convDense, [1, num_trades, 1])), 2)

            # Merge all streams
            mergedDense = tf.concat((convDense, sumTrades), axis=2)

            # Output layers
            ## Init
            self.longInit = slim.fully_connected(mergedDense, 2, activation_fn=tf.nn.softmax)
            self.longSize = tf.reshape(slim.fully_connected(mergedDense, 1, activation_fn=tf.nn.sigmoid), (-1,))

            self.shortInit = slim.fully_connected(mergedDense, 2, activation_fn=tf.nn.softmax)
            self.shortSize = tf.reshape(slim.fully_connected(mergedDense, 1, activation_fn=tf.nn.sigmoid), (-1,))

            ## Close
            self.tradeClose = slim.fully_connected(tradeDense, 2, activation_fn=tf.nn.softmax)

            ## Value
            self.value = tf.reshape(slim.fully_connected(mergedDense, 1, activation_fn=None, biases_initializer=None),
                                    (-1,))


class Worker():
    def __init__(self, env, window_size, work_id, trainer, model_path):
        self.name = "worker_" + str(work_id)
        self.number = work_id
        self.model_path = model_path
        self.trainer = trainer
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_principals = []
        self.num_trades = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.local_AC = AC_Network(window_size, self.name, trainer)
        self.env = env

    def test(self, temp=0.5):
        self.env.test()
        t = 0
        s = self.env.get_state()
        while not t:
            s[0] = stock_relativized_state(s[0])

            lI, lS, sI, sS, tC, v = sess.run([
                self.local_AC.longInit,
                self.local_AC.longSize,
                self.local_AC.shortInit,
                self.local_AC.shortSize,
                self.local_AC.tradeClose,
                self.local_AC.value],
                feed_dict={self.local_AC.price_input: np.expand_dims(s[0][:, :4], 0),
                           self.local_AC.volume_input: np.expand_dims(np.expand_dims(s[0][:, 4], 0).T, 0),
                           self.local_AC.trade_input: np.reshape(s[1], (1, -1, 2))})

            a = [[1 * (lI[0][0][1] > temp), lS[0], 1 * (sI[0][0][1] > temp), sS[0]],
                 map(lambda x: 1 * (x[1] > temp), tC[0])]

            r, t, s = self.env.step(a)


if __name__ == '__main__':
    window_size = 24
    ticker = sys.argv[1]
    data_path = '/tmp/ticker_data_{}.csv'.format(ticker)
    results_path = '/tmp/tradelog_{}.csv'.format(ticker)

    share = Share(ticker)
    try:
        data = share.get_historical('2016-04-25', '2017-04-29')
        data = [[d['Date'],float(d['Open']), float(d['High']), float(d['Low']), float(d['Close']), int(d['Volume'])] for d in data]
        env = StockTradeSimulator([d[1:] for d in data], 0, window_size)
        pd.DataFrame(data).to_csv(data_path)
    except:
        with open(data_path, 'w') as file:
            file.writelines('Bad Ticker')
    try:
        tf.reset_default_graph()

        with tf.device("/cpu:0"):
            trainer = tf.train.AdamOptimizer(learning_rate=1e-6)

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            worker = Worker(env, window_size, 0, trainer, 0)

            sess.run(tf.global_variables_initializer())

            worker.test(0.5)


        with open(results_path, 'w') as file:
            file.writelines('direction, size, init_index, close_index')
            for trade in worker.env.trade_log:
                file.writelines('\n'+','.join([str(t) for t in trade]))
    except:
        with open(results_path, 'w') as file:
            file.writelines('direction, size, init_index, close_index')
