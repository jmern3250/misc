import tensorflow as tf
import PIL 
from PIL import Image
import scipy.io as sio
import glob 
import argparse

from model import * 
from discriminator import discriminator

import numpy as np
from numpy import matlib
import math
import timeit

def main(args):
    X_train, Y_train = load_data(args.data)
    Y_train_ = np.random.shuffle(Y_train)

    if args.GPU == 0:
        config = tf.ConfigProto(
                device_count = {'GPU': 0}
                )
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False

    tf.reset_default_graph()
    if args.data == 0:
        X = tf.placeholder(tf.float32, [None, 480, 640, 3])
        Y = tf.placeholder(tf.float32, [None, 480, 640, 1])
    elif args.data == 1:
        X = tf.placeholder(tf.float32, [None, 245, 437, 3])
        Y = tf.placeholder(tf.float32, [None, 245, 437, 1])
        X_ = tf.placeholder(tf.float32, [None, 245, 437, 3])
        Y_ = tf.placeholder(tf.float32, [None, 245, 437, 1])
    is_training = tf.placeholder(tf.bool)
    
    with tf.variable_scope('Encoder') as enc: 
        latent_y = encoder(X, is_training, args.data)
    with tf.variable_scope('Decoder') as dec:
        output = decoder(latent_y, is_training, args.data)
    with tf.variable_scope('Discriminator') as dis: 
        D_x = discriminator(X, output) #from generated result
    with tf.variable_scope(dis, reuse=True): 
        D_y = discriminator(X, Y)
        D_hat = discriminator(X_, Y_)

    critic_loss = tf.reduce_mean(D_x) - tf.reduce_mean(D_y)

    d = tf.gradeients(D_hat, [X_, Y_])
    # import pdb; pdb.set_trace()
    dx = d[0]
    dy = d[1]
    gradient_penalty = tf.sqrt(tf.reduce_sum(tf.square(dx),axis=1) + tf.reduce_sum(tf.square(dy),axis=1))
    gradient_penalty = 10.0*tf.reduce_mean(tf.square(gradient_penalty - 1.0))
    # gradient_penalty = 10.0*tf.reduce_mean((tf.norm(tf.gradients(critic_loss_, X_), ord='euclidean', axis=[1,2])-1)**2)
    

    disc_loss = critic_loss + gradient_penalty  
    
    gen_disc_loss  = -tf.reduce_mean(D_x)
    l1_loss = 100.0*l1_norm(output-Y)

    gen_loss = gen_disc_loss + l1_loss

    tf.summary.scalar('disc_loss\critic_loss', critic_loss)
    tf.summary.scalar('disc_loss\gradient_penalty', gradient_penalty)
    tf.summary.scalar('disc_loss\disc_loss', disc_loss)

    tf.summary.scalar('gen_loss\disc_loss', gen_disc_loss)
    tf.summary.scalar('gen_loss\l1_loss', l1_loss)
    tf.summary.scalar('gen_loss\gen_loss', gen_loss)
    

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Encoder')
    dec_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Decoder')
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Discriminator')
    disc_weights = []
    for var in disc_vars:
        if 'kernel' in var.name: 
            disc_weights.append(var)


    gen_optimizer = tf.train.AdamOptimizer(learning_rate=args.rate,beta1=0.0,beta2=0.9,name='gen_adam')
    disc_optimizer = tf.train.AdamOptimizer(learning_rate=args.rate,beta1=0.0,beta2=0.9,name='disc_adam')
    with tf.control_dependencies(extra_update_ops):
        train_discriminator = disc_optimizer.minimize(disc_val, var_list=[disc_vars])
        train_generator = gen_optimizer.minimize(mean_loss,var_list=[enc_vars, dec_vars])
    
    sess = tf.Session(config=config)
    enc_saver = tf.train.Saver(var_list=enc_vars)
    dec_saver = tf.train.Saver(var_list=dec_vars)
    disc_saver = tf.train.Saver(var_list=disc_vars)

    savers = [enc_saver, dec_saver, disc_saver]

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tb',sess.graph)

    sess.run(tf.global_variables_initializer())

    _ = run_model(sess, X, Y, X_, Y_, is_training, disc_val, mean_loss, X_train, Y_train, savers,  
              epochs=args.epochs, batch_size=args.batch_size, print_every=10,
              disc_training=train_discriminator, gen_training=train_generator, 
              plot_losses=False, writer=writer, sum_vars=merged)

    model_name = './models/final_model'
    enc_saver.save(sess, model_name+'_enc')
    dec_saver.save(sess, model_name+'_dec')
    disc_saver.save(sess, model_name+'_disc')

def load_data(data_idx, num=None):
    if data_idx == 0:
        mat_contents = sio.loadmat('./data/NYU_data.mat')
        X_train = mat_contents['images'].transpose([3,0,1,2])/255.0
        Y_train = mat_contents['depths'].transpose([2,0,1])
        Y_train = Y_train.reshape([1449,480,640,1])
        Y_train /= Y_train.max()
        Y_train -= 1
        Y_train *= -1.0
    elif data_idx == 1: 
        if num is not None: 
            TRAIN=num
        else:
            TRAIN=1963 

        X_train = np.zeros([TRAIN, 245, 437, 3])
        Y_train = np.zeros([TRAIN, 245, 437, 1])

        i = 0
        for filename in sorted(glob.glob('../data/AirSim/Scene/*.jpg')): 
            im=Image.open(filename)
            X_train[i,:,:,:] = (np.array(im)[:,:,:]/255.0*2.0)-1.0
            i += 1
            if i == TRAIN: 
                break

        i = 0
        for filename in sorted(glob.glob('../data/AirSim/Depth/*.jpg')): 
            im=Image.open(filename)
            img_array = np.array(im)
            if img_array.ndim == 3:
                img_array = img_array[:,:,0]
            Y_train[i,:,:,0] = (img_array[:,:]/255.0*2.0)-1.0
            i += 1
            if i == TRAIN:
                break 

    return X_train, Y_train

def run_model(session, X, Y, X_, Y_, is_training, disc_val, loss_val, Xd, Yd, savers, 
              epochs=1, batch_size=64, print_every=100,
              disc_training=None, gen_training=None, 
              plot_losses=False, writer=None, sum_vars=None):

    enc_saver, dec_saver, disc_saver = savers
    
    # shuffle indicies
    gen_train_indicies = np.arange(Xd.shape[0])
    disc_train_indicies = np.arange(Xd.shape[0])
    n_critic = 5
    disc_train_indicies = np.tile(disc_train_indicies, n_critic)

    np.random.shuffle(gen_train_indicies)
    np.random.shuffle(disc_train_indicies)
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    gen_variables = [loss_val, gen_training]
    disc_variables = [disc_val, disc_training]
    if writer is not None: 
        gen_variables.append(sum_vars)
        disc_variables.append(sum_vars)

    # counter 
    iter_cnt = 0
    for e in range(epochs):
        losses = []
        
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            gen_start_idx = (i*batch_size)%Xd.shape[0]
            gen_idx = gen_train_indicies[start_idx:start_idx+batch_size]
            for j in range(n_critic):
                disc_start_idx = (i*n_critic + j)*batch_size%Xd.shape[0] 
                disc_idx = disc_train_indicies[start_idx:start_idx+batch_size]
                disc_feed_dict = {X: Xd[disc_idx,:],
                            Y: Yd[disc_idx,:],
                            X_: (Xd[disc_idx,:]+Xd[gen_idx,:])/2.0,
                            Y_: (Yd[disc_idx,:]+Yd[gen_idx,:])/2.0,
                            is_training: True}
                if writer is not None:
                    d_loss, _, _ = session.run(disc_variables, feed_dict=disc_feed_dict)
                else:
                    d_loss, _ = session.run(disc_variables, feed_dict=disc_feed_dict)
            
            # create a feed dictionary for this batch
            gen_feed_dict = {X: Xd[gen_idx,:],
                            Y: Yd[gen_idx,:],
                            is_training: True}
            
            # get batch size
            actual_batch_size = Yd[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            if writer is not None:
                d_loss, _, _, _ = session.run(disc_variables, feed_dict=disc_feed_dict)
                loss, _, summary = session.run(gen_variables,feed_dict=gen_feed_dict)
                writer.add_summary(summary, iter_cnt)
            else:
                d_loss, _, _ = session.run(disc_variables, feed_dict=disc_feed_dict)
                loss, _ = session.run(gen_variables,feed_dict=gen_feed_dict)
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            
            #save checkpoint
            if (e+1 % 10) == 0:
                enc_saver.save(session, './models/int_model_enc', global_step=e)
                dec_saver.save(session, './models/int_model_dec', global_step=e)
                disc_saver.save(session, './models/int_model_disc', global_step=e)

            # print every now and then
            if (iter_cnt % print_every) == 0:
                print("Iteration %r: with generator loss = %r and discriminator loss = %r " % (iter_cnt,loss,d_loss))
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
	norm = tf.reduce_mean(X)
	return norm 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CNN translation for given arguments')
    parser.add_argument('data', type=int) #0:NYU, 1:Airsim
    parser.add_argument('epochs', type=int)
    parser.add_argument('batch_size', type=int) 
    parser.add_argument('rate', type=float) 
    # parser.add_argument('lam', type=float) 
    parser.add_argument('GPU', type=int) 
    args = parser.parse_args()
    main(args)
