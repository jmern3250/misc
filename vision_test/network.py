import tensorflow as tf 
import logger

import numpy as np 
import pickle
import cv2
import copy

import matplotlib.pyplot as plt

#DEFAULT LOGGER PARAMETERS
log_strs = ['stdout', 'log', 'csv', 'tensorboard']
logger.configure(dir='./log/', format_strs=log_strs)
# logger.set_level(logger.INFO)

N_NAMES = 127
N_LVLS = 76

with open('./name_dict.p', 'rb') as f:
    NAME_DICT = pickle.load(f)

class OCRNet:
    def __init__(self, h, w, dtype=tf.float32, GPU=False):
        self.h = h
        self.w = w
        self.dtype = dtype
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, [None, self.h, self.w, 3], name='X')
            self.NAME = tf.placeholder(tf.int32, [None], name='NAME')
            self.LVL = tf.placeholder(tf.int32, [None], name='LVL')
            self.RED = tf.placeholder(tf.int32, [None], name='RED')
            self.GOLD = tf.placeholder(tf.int32, [None], name='GOLD')
        if not GPU:
            config = tf.ConfigProto(
                    device_count = {'GPU': 0} 
                    )
        else:
            print("GPU Available: ", tf.test.is_gpu_available())
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config,graph=self.graph)

    def build_ocr_net(self):
        with self.graph.as_default():
            self.Z = self._build_base_net(self.X, 'base')
            self.name = self._build_head_net(self.Z, N_NAMES, 'name')
            self.lvl = self._build_head_net(self.Z, N_LVLS, 'lvl')
            self.red = self._build_head_net(self.Z, 8, 'red')
            self.gold = self._build_head_net(self.Z, 8, 'gold')

            self.name_idx = tf.argmax(self.name, axis=1)
            self.lvl_idx = tf.argmax(self.lvl, axis=1)
            self.red_idx = tf.argmax(self.red, axis=1)
            self.gold_idx = tf.argmax(self.gold, axis=1)

            self._base_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='base')
            self.base_saver = tf.train.Saver(var_list=self._base_vars)
            self._name_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='name')
            self.name_saver = tf.train.Saver(var_list=self._name_vars)
            self._lvl_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lvl')
            self.lvl_saver = tf.train.Saver(var_list=self._lvl_vars)
            self._red_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='red')
            self.red_saver = tf.train.Saver(var_list=self._red_vars)
            self._gold_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gold')
            self.gold_saver = tf.train.Saver(var_list=self._gold_vars)

    def _build_base_net(self, input_image, name='base'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            input_tensor = tf.layers.conv2d(
                                input_image,
                                8,
                                (5,5),
                                strides=(2, 2),
                                padding='valid',
                                activation=tf.nn.leaky_relu,
                                use_bias=True,
                                trainable=True,
                                name='conv_0'
                            )
            for i in range(3):
                input_tensor = tf.layers.conv2d(
                                input_tensor,
                                16*2**i,
                                (3,3),
                                strides=(2, 2),
                                padding='valid',
                                activation=tf.nn.leaky_relu,
                                use_bias=True,
                                trainable=True,
                                name='conv_' + str(i + 1)
                            )
        return input_tensor

    def _build_head_net(self, input_tensor, output_dim, name='head'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for i in range(2):
                _, h, w, _ = input_tensor.shape
                input_tensor = tf.layers.conv2d(
                                input_tensor,
                                64*2**i,
                                (3,3),
                                strides=(2, 2),
                                padding='valid',
                                activation=tf.nn.leaky_relu,
                                use_bias=True,
                                trainable=True,
                                name='conv_' + str(i)
                            )
            input_tensor = tf.layers.flatten(input_tensor)
            input_tensor = tf.layers.dense(
                                input_tensor,
                                output_dim*4,
                                activation=tf.nn.leaky_relu,
                                use_bias=False,
                                trainable=True,
                                name='dense_0'
                            )
            input_tensor = tf.layers.dense(
                                input_tensor,
                                output_dim*2,
                                activation=tf.nn.leaky_relu,
                                use_bias=False,
                                trainable=True,
                                name='dense_1'
                            )
            logits = tf.layers.dense(
                                input_tensor,
                                output_dim,
                                activation=None,
                                use_bias=False,
                                trainable=True,
                                name='logits'
                            )

        return logits

    def show_image(self, im_array):
        im_array += 1.
        im_array *= 255/2
        im_array = im_array.astype(np.uint8)
        bgr = cv2.cvtColor(im_array, cv2.COLOR_YCrCb2BGR)
        cv2.imshow('frame', bgr)
        cv2.waitKey(30)

    def restore_model(self, filename=None):
        with self.graph.as_default():
            self.session.run(tf.global_variables_initializer())
            self.initialized = True
            if filename is not None:
                self.base_saver.restore(self.session, filename + '_base')
                try:
                    self.name_saver.restore(self.session, filename + '_name')
                except:
                    print('Name Model Incorrect or not Found. Skipping...')
                try:
                    self.lvl_saver.restore(self.session, filename + '_lvl')
                except:
                    print('Level Model Incorrect or not Found. Skipping...')
                try:
                    self.red_saver.restore(self.session, filename + '_red')
                except:
                    print('Red Model Incorrect or not Found. Skipping...')
                try:
                    self.gold_saver.restore(self.session, filename + '_gold')
                except:
                    print('Gold Model Incorrect or not Found. Skipping...')

    def save_model(self, filename):
        with self.graph.as_default():
            self.base_saver.save(self.session, filename + '_base')
            self.name_saver.save(self.session, filename + '_name')
            self.lvl_saver.save(self.session, filename + '_lvl')
            self.red_saver.save(self.session, filename + '_red')
            self.gold_saver.save(self.session, filename + '_gold')

    def ocr_train(self, train_file, learning_rate,
                epochs, model_name, weight=0.1, checkpoint=None):
        with self.graph.as_default():
            name_labels = tf.one_hot(self.NAME, N_NAMES)
            lvl_labels = tf.one_hot(self.LVL, N_LVLS)
            red_labels = tf.one_hot(self.RED, 8)
            gold_labels = tf.one_hot(self.GOLD, 8)

            name_loss = tf.losses.softmax_cross_entropy(name_labels, self.name)
            lvl_loss = tf.losses.softmax_cross_entropy(lvl_labels, self.lvl)
            red_loss = tf.losses.softmax_cross_entropy(red_labels, self.red)
            gold_loss = tf.losses.softmax_cross_entropy(gold_labels, self.gold)
            
            total_loss = lvl_loss + weight*(name_loss + red_loss + gold_loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.90,
                                            beta2=0.999, epsilon=1e-6, name='adam')
            grads_vars = optimizer.compute_gradients(total_loss) 
            grads = [grad for grad, _ in grads_vars]
            varbs = [varb for _, varb in grads_vars]
            clipped_grads, grad_norm = tf.clip_by_global_norm(
                                                grads,
                                                1.,
                                                name='clipped_grads'
                                                )
            clipped_grad_norm = tf.global_norm(clipped_grads)
            clipped_grads_vars = list(zip(clipped_grads, varbs))
            train_op = optimizer.apply_gradients(clipped_grads_vars)
           
            tf.summary.FileWriter('./log/',
                                      self.graph)
        self.restore_model(filename=checkpoint)

        train_variables = ( train_op,
                            name_loss, 
                            lvl_loss, 
                            red_loss, 
                            gold_loss,
                            total_loss,
                            grad_norm,
                            clipped_grad_norm
                            )
        with open(train_file, 'rb') as f:
            train_tuple = pickle.load(f)
        train_data = train_tuple[0]
        name_dict = train_tuple[2]

        n_sets = len(train_data)
        train_idxs = np.arange(n_sets)
        itr = 0
        ITRPRINT = 10
        for epoch in range(epochs):
            logger.log("********** Epoch %i ************"%epoch)
            np.random.shuffle(train_idxs)
            for i in train_idxs:
                if (itr%ITRPRINT) == 0:
                    n_losses = []
                    l_losses = []
                    r_losses = []
                    g_losses = []
                    tot_losses = []
                    grads = []
                    cgrads = []
                train_set = train_data[i]
                images = np.stack(train_set['imgs']).astype(np.float32)
                images /= 255.0
                images *= 2.0
                images -= 1.0

                valid = True
                n_images = len(images)
                names = np.zeros(n_images, dtype=np.int32)
                for j in range(n_images):
                    name = train_set['names'][j]
                    try:
                        idx = name_dict[name]
                        names[j] = idx
                    except:
                        valid = False

                if valid:
                    lvls = np.array(train_set['lvls'], dtype=np.int32)
                    reds = train_set['red']
                    golds = train_set['gold']
                    feed_dict = {
                                 self.X:images,
                                 self.NAME:names,
                                 self.LVL:lvls,
                                 self.RED:reds,
                                 self.GOLD:golds
                                }
                    _, n_loss, l_loss, r_loss, g_loss, tot_loss, grad, cgrad  = self.session.run(train_variables, feed_dict=feed_dict)
                    n_losses.append(n_loss)
                    l_losses.append(l_loss)
                    r_losses.append(r_loss)
                    g_losses.append(g_loss)
                    tot_losses.append(tot_loss)
                    grads.append(grad)
                    cgrads.append(cgrad)
                    if (itr%ITRPRINT) == 0:
                        logger.log('##### Iteration %i #####'%itr)
                        logger.log('(Epoch: %i)'%epoch)
                        logger.record_tabular('1. Mean Name Loss', np.mean(n_losses))
                        logger.record_tabular('2. Mean Lvl Loss', np.mean(l_losses))
                        logger.record_tabular('3. Mean Red Loss', np.mean(r_losses))
                        logger.record_tabular('4. Mean Gold Loss', np.mean(g_losses))
                        logger.record_tabular('5. Mean Total Loss', np.mean(tot_losses))
                        logger.record_tabular('6. Mean Gradient Norm', np.mean(grads))
                        logger.record_tabular('7. Mean Clipped Gradient', np.mean(cgrads))
                        logger.dump_tabular()
                    itr += 1
            self.save_model(model_name + '_' + str(epoch))

    def ocr_test(self, test_file, checkpoint):

        self.restore_model(filename=checkpoint)

        test_variables = (self.name_idx, 
                        self.lvl_idx, 
                        self.red_idx, 
                        self.gold_idx)
        with open(test_file, 'rb') as f:
            test_tuple = pickle.load(f)
        test_data = test_tuple[0]
        name_dict = test_tuple[2]

        n_name_errors = 0
        n_lvl_errors = 0
        n_red_errors = 0
        n_gold_errors = 0
        n_total = 0
        for i, test_set in enumerate(test_data):
            images = np.stack(test_set['imgs']).astype(np.float32)
            images /= 255.0
            images *= 2.0
            images -= 1.0

            valid = True
            n_images = len(images)
            true_names = np.zeros(n_images, dtype=np.int32)
            for j in range(n_images):
                name = test_set['names'][j]
                try:
                    idx = name_dict[name]
                    true_names[j] = idx
                except:
                    valid = False

            if valid:
                true_lvls = np.array(test_set['lvls'], dtype=np.int32)
                true_reds = test_set['red']
                true_golds = test_set['gold']
                feed_dict = {self.X:images}
                names, lvls, reds, golds  = self.session.run(test_variables, feed_dict=feed_dict)
                
                reds = np.round(reds).flatten()
                golds = np.round(golds).flatten()
                n_name_errors += np.sum(names != true_names)
                n_lvl_errors += np.sum(lvls != true_lvls)
                n_red_errors += np.sum(reds != true_reds)
                n_gold_errors += np.sum(golds != true_golds)
                n_total += n_images

            print("Test set %i complete" % i)
        print("Name Error Rate: ", n_name_errors*100.0/n_total)
        print("Lvl Error Rate: ", n_lvl_errors*100.0/n_total)
        print("Red Error Rate: ", n_red_errors*100.0/n_total)
        print("Gold Error Rate: ", n_gold_errors*100.0/n_total)
    
    def encode_image(self, image):    
        run_variables = (self.name_idx, 
                        self.lvl_idx, 
                        self.red_idx, 
                        self.gold_idx)
        image /= 255.0
        image *= 2.0
        image -= 1.0
        feed_dict = {self.X:np.expand_dims(image, axis=0)}
        name_idx, lvl, red, gold  = self.session.run(run_variables, feed_dict=feed_dict)
        name = NAME_DICT[name_idx[0]]
        return name, lvl[0], red[0], gold[0]


if __name__ == "__main__":
    graph = OCRNet(137, 217, GPU=True)
    graph.build_ocr_net()
    # graph.ocr_train('./data/train_data.p', 1e-3,
    #             100, './models/net_0', weight=0.1, checkpoint=None)
    # graph.ocr_test('./data/train_data.p', './models/net_0_41')

    # graph.ocr_train('./data/train_data.p', 1e-4,
    #             100, './models/net_1', weight=0.1, checkpoint='./models/net_0_41')
    # graph.ocr_test('./data/train_data.p', './models/net_1_15')

    test_img = plt.imread('./data/img0.png')[...,:3]*255.
    # plt.imshow(test_img)
    # plt.show()
    # import pdb; pdb.set_trace()
    graph.restore_model('./models/net_1_15')
    test = graph.encode_image(test_img)
    [print(itm) for itm in test]