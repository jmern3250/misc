import tensorflow as tf
import pickle
import argparse

import numpy as np
import math

def main(args):
    Xd, Yd = load_data()
    # import pdb; pdb.set_trace()
    if args.GPU == 0:
        config = tf.ConfigProto(
                device_count = {'GPU': 0}
                )
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 28])
    Y = tf.placeholder(tf.float32, [None, 1])
    reg_loss_fn = tf.contrib.layers.l2_regularizer(0.001)
    output = MLP(X,reg_loss_fn)
    # import pdb; pdb.set_trace()

    l2_loss = tf.reduce_mean((output-Y)**2)
    l1_loss = tf.reduce_mean(tf.sqrt((output-Y)**2))
    reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = tf.contrib.layers.apply_regularization(reg_loss_fn, reg_vars)
    mean_loss = 0.9*l2_loss + 0.1*l1_loss + 1e-7*reg_loss
    tf.summary.scalar('loss', mean_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.rate)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_full = optimizer.minimize(mean_loss)  
    
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tb',sess.graph)

    sess.run(tf.global_variables_initializer())
    _ = run_model(sess, X, Y, mean_loss, Xd, Yd, 
              epochs=args.epochs, batch_size=args.batch_size, print_every=10, 
              training=train_full, plot_losses=False,
              writer=writer, sum_vars=merged)

    model_name = './MLP/MLP'
    saver.save(sess, model_name)

def load_data():
    data = np.loadtxt('./Data/extdata.csv', delimiter=',')
    Y = data[:,0].reshape([-1,1])
    X = data[:,1:]
    return X, Y 


def run_model(session, X, Y, loss_val, Xd, Yd, 
              epochs=1, batch_size=64, print_every=10000,
              training=None, plot_losses=False,writer=None, sum_vars=None):
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_val, training]
    if writer is not None: 
        variables.append(sum_vars)

    # counter 
    iter_cnt = 0
    for e in range(epochs):
        losses = []
        
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         Y: Yd[idx,:]}
            # get batch size
            actual_batch_size = Yd[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            if writer is not None:
                # import pdb; pdb.set_trace()
                loss, _, summary = session.run(variables,feed_dict=feed_dict)
                writer.add_summary(summary, iter_cnt)
            else:
                loss, _ = session.run(variables,feed_dict=feed_dict)
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            
            # print every now and then
            if (iter_cnt % print_every) == 0:
                print("Iteration %r: with minibatch training loss = %r " % (iter_cnt,loss))
            iter_cnt += 1

        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {1}, Overall loss = {0:.3g}"\
              .format(total_loss,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss

def l1_norm(X):
    # X = tf.sqrt(X**2)
    X = tf.abs(X)
    norm = tf.reduce_sum(X)
    return norm

def MLP(X, regularizer=None):
    h1 = tf.layers.dense(X, 64, 
                        activation=tf.nn.relu, 
                        use_bias=True,
                        kernel_regularizer=regularizer,
                        name='h1'
                        )
    h2 = tf.layers.dense(h1, 128, 
                        activation=tf.nn.relu, 
                        use_bias=True,
                        kernel_regularizer=regularizer,
                        name='h2'
                        )
    h3 = tf.layers.dense(h2, 256, 
                        activation=tf.nn.relu, 
                        use_bias=True,
                        kernel_regularizer=regularizer,
                        name='h3'
                        )
    h4 = tf.layers.dense(h3, 512, 
                        activation=tf.nn.relu, 
                        use_bias=True,
                        kernel_regularizer=regularizer,
                        name='h4'
                        )
    h5 = tf.layers.dense(h4, 64, 
                        activation=tf.nn.relu, 
                        use_bias=True,
                        kernel_regularizer=regularizer,
                        name='h5'
                        )
    output = tf.layers.dense(h5, 1, 
                        activation=None, 
                        use_bias=True,
                        kernel_regularizer=regularizer,
                        name='output'
                        )
    return output 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CNN translation for given arguments')
    parser.add_argument('epochs', type=int)
    parser.add_argument('batch_size', type=int) 
    parser.add_argument('rate', type=float) 
    # parser.add_argument('lam', type=float) 
    parser.add_argument('GPU', type=int) 
    args = parser.parse_args()
    main(args)
