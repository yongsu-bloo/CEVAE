#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
import os, random
import edward as ed
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # shut up tensorflow
from edward.models import Bernoulli, Normal, BernoulliWithSigmoidProbs
# from progressbar import ETA, Bar, Percentage, ProgressBar

from datasets import IHDP, TWINS, JOBS
from evaluation import Evaluator
import numpy as np
import time
from scipy.stats import sem

from utils import fc_net, get_y0_y1, write_results
import arguments

args = arguments.parse_args()
exp_name = args.exp_name

task = args.task
test_only = args.evaluate

args.true_post = True
pnoise_type = args.pnoise
pnoise_size = args.pn_size
pnoise_scale = args.pn_scale

data_path = args.data_path
save_model = args.save_model

if not save_model:
    save_model = 'models/' + exp_name

if not os.path.exists(save_model):
    os.mkdir(save_model)
load_model = args.load_model


data_pref = '_'.join([ str(i) for i in [pnoise_type, pnoise_size, pnoise_scale, ""]]) if pnoise_type is not None else ""

if task == 'ihdp':
    dataset = IHDP(replications=args.reps, data_pref=data_pref, path_data=data_path)
elif task == 'twins':
    dataset = TWINS(replications=args.reps, data_pref=data_pref)
elif task == 'jobs':
    dataset = JOBS(replications=args.reps, data_pref=data_pref)
# dimx = 25
scores = np.zeros((args.reps, 3))
scores_test = np.zeros((args.reps, 3))

M = None  # batch size during training
d = args.latent_dim  # latent dimension
lamba = args.lamba  # weight decay
nh, h = args.nh, args.h_dim  # number and size of hidden layers
batch_size = args.batch_size
epochs = args.epochs

lr = args.lr

drop_ratio = args.drop_ratio
drop_type = args.drop_type

arg_info = {
    "exp_name" : exp_name,
    "latent_dim" : d,
    "nh" : nh,
    "hidden_dim" : h,
    "batch_size" : batch_size,
    "lr" : lr,
    "epochs" : epochs,
    "lamba" : lamba,

}

if args.noise > 0:
    noises = [0.1*i for i in range(args.noise)]
else:
    noises = [0.]
num_seeds = 12

for gnoise in noises:
    for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
        print('\nReplication {}/{}'.format(i + 1, args.reps))
        # print train
        if task == 'jobs':
            (xtr, ttr, ytr), etr = train
            (xva, tva, yva), eva = valid
            (xte, tte, yte), ete = test
            evaluator_test = Evaluator(yte, tte, e=ete, task=task)
        else:
            (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
            (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
            (xte, tte, yte), (y_cfte, mu0te, mu1te) = test
            evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te, task=task)
        num_train = len(xtr) + len(xva)
        num_test = len(xte)
        # reorder features with binary first and continuous after
        perm = binfeats + contfeats
        xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

        xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate([ytr, yva], axis=0)
        if task == 'jobs':
            ealltr = np.concatenate([etr, eva], axis=0)
            evaluator_train = Evaluator(yalltr, talltr, e=ealltr, task=task)
        else:
            evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                        mu0=np.concatenate([mu0tr, mu0va], axis=0), mu1=np.concatenate([mu1tr, mu1va], axis=0), task=task)

        # zero mean, unit variance for y during training
        if task == 'ihdp':
            ym, ys = np.mean(ytr), np.std(ytr)
            ytr, yva = (ytr - ym) / ys, (yva - ym) / ys

        best_logpvalid = - np.inf
        with tf.Graph().as_default():
            sess = tf.InteractiveSession()

            ed.set_seed(1)
            np.random.seed(1)
            tf.set_random_seed(1)

            x_ph_bin = tf.placeholder(tf.float32, [M, len(binfeats)], name='x_bin')  # binary inputs
            x_ph_cont = tf.placeholder(tf.float32, [M, len(contfeats)], name='x_cont')  # continuous inputs
            t_ph = tf.placeholder(tf.float32, [M, 1])
            y_ph = tf.placeholder(tf.float32, [M, 1])

            x_ph = tf.concat([x_ph_bin, x_ph_cont], 1)
            activation = tf.nn.elu

            """Model Building"""
            # CEVAE model (decoder)
            # p(z)
            mu_z = tf.zeros([tf.shape(x_ph)[0], d])
            sigma_z = tf.ones([tf.shape(x_ph)[0], d])
            z = Normal(loc=mu_z, scale=sigma_z)

            # p(x|z)
            hx = fc_net(z, (nh - 1) * [h], [], 'px_z_shared', lamba=lamba, activation=activation)
            logits = fc_net(hx, [h], [[len(binfeats), None]], 'px_z_bin'.format(i + 1), lamba=lamba, activation=activation)
            x1 = Bernoulli(logits=logits, dtype=tf.float32, name='bernoulli_px_z')

            mu, sigma = fc_net(hx, [h], [[len(contfeats), None], [len(contfeats), tf.nn.softplus]], 'px_z_cont', lamba=lamba,
                               activation=activation)
            x2 = Normal(loc=mu, scale=sigma, name='gaussian_px_z')

            # p(t|z)
            logits = fc_net(z, [h], [[1, None]], 'pt_z', lamba=lamba, activation=activation)
            t = Bernoulli(logits=logits, dtype=tf.float32)

            # p(y|t,z)
            mu2_t0 = fc_net(z, nh * [h], [[1, None]], 'py_t0z', lamba=lamba, activation=activation)
            mu2_t1 = fc_net(z, nh * [h], [[1, None]], 'py_t1z', lamba=lamba, activation=activation)
            if task != 'ihdp':
                y = BernoulliWithSigmoidProbs(logits=t * mu2_t1 + (1. - t) * mu2_t0, dtype=tf.float32)
            else:
                y = Normal(loc=t * mu2_t1 + (1. - t) * mu2_t0, scale=tf.ones_like(mu2_t0))
            # CEVAE variational approximation (encoder)
            # q(t|x)
            logits_t = fc_net(x_ph, [d], [[1, None]], 'qt', lamba=lamba, activation=activation)
            qt = Bernoulli(logits=logits_t, dtype=tf.float32)
            # q(y|x,t)
            hqy = fc_net(x_ph, (nh - 1) * [h], [], 'qy_xt_shared', lamba=lamba, activation=activation)
            mu_qy_t0 = fc_net(hqy, [h], [[1, None]], 'qy_xt0', lamba=lamba, activation=activation)
            mu_qy_t1 = fc_net(hqy, [h], [[1, None]], 'qy_xt1', lamba=lamba, activation=activation)
            if task != 'ihdp':
                asdf = tf.nn.sigmoid(qt * mu_qy_t1 + (1. - qt) * mu_qy_t0)
                fdsa = qt * mu_qy_t1 + (1. - qt) * mu_qy_t0
                qy = BernoulliWithSigmoidProbs(logits=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, dtype=tf.float32)
            else:
                qy = Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))
            # q(z|x,t,y)
            inpt2 = tf.concat([x_ph, qy], 1)
            hqz = fc_net(inpt2, (nh - 1) * [h], [], 'qz_xty_shared', lamba=lamba, activation=activation)
            muq_t0, sigmaq_t0 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0', lamba=lamba,
                                       activation=activation)
            muq_t1, sigmaq_t1 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt1', lamba=lamba,
                                       activation=activation)
            muq = qt * muq_t1 + (1. - qt) * muq_t0
            sigmaq = qt * sigmaq_t1 + (1. - qt) * sigmaq_t0

            qz = Normal(loc=muq, scale=sigmaq)

            # Create data dictionary for edward
            data = {x1: x_ph_bin, x2: x_ph_cont, y: y_ph, qt: t_ph, t: t_ph, qy: y_ph}

            # sample posterior predictive for p(y|z,t)
            y_post = ed.copy(y, {z: qz, t: t_ph}, scope='y_post')
            # crude approximation of the above
            y_post_mean = ed.copy(y, {z: qz.mean(), t: t_ph}, scope='y_post_mean')
            # construct a deterministic version (i.e. use the mean of the approximate posterior) of the lower bound
            # for early stopping according to a validation set
            y_post_eval = ed.copy(y, {z: qz.mean(), qt: t_ph, qy: y_ph, t: t_ph}, scope='y_post_eval')
            x1_post_eval = ed.copy(x1, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='x1_post_eval')
            x2_post_eval = ed.copy(x2, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='x2_post_eval')
            t_post_eval = ed.copy(t, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='t_post_eval')
            # losses
            # yt_post_loss = tf.reduce_mean(tf.reduce_sum(y_post_eval.log_prob(y_ph) + t_post_eval.log_prob(t_ph), axis=1))
            # x1_post_loss = tf.reduce_mean(tf.reduce_sum(x1_post_eval.log_prob(x_ph_bin), axis=1))
            # x2_post_loss = tf.reduce_mean(tf.reduce_sum(x2_post_eval.log_prob(x_ph_cont), axis=1))
            # z_qz_loss = tf.reduce_mean(tf.reduce_sum(z.log_prob(qz.mean()) - qz.log_prob(qz.mean()), axis=1))

            logp_valid = tf.reduce_mean(tf.reduce_sum(y_post_eval.log_prob(y_ph) + t_post_eval.log_prob(t_ph), axis=1) +
                                        tf.reduce_sum(x1_post_eval.log_prob(x_ph_bin), axis=1) +
                                        tf.reduce_sum(x2_post_eval.log_prob(x_ph_cont), axis=1) +
                                        tf.reduce_sum(z.log_prob(qz.mean()) - qz.log_prob(qz.mean()), axis=1))

            inference = ed.KLqp({z: qz}, data)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            inference.initialize(optimizer=optimizer)

            saver = tf.train.Saver(tf.contrib.slim.get_variables())
            tf.global_variables_initializer().run()

            n_epoch, n_iter_per_epoch, idx = epochs, max(10 * int(xtr.shape[0] / batch_size), 1), np.arange(xtr.shape[0])

            # dictionaries needed for evaluation
            tr0, tr1 = np.zeros((xalltr.shape[0], 1)), np.ones((xalltr.shape[0], 1))
            tr0t, tr1t = np.zeros((xte.shape[0], 1)), np.ones((xte.shape[0], 1))

            """make noise"""
            if gnoise > 0:
                gnoise_train = np.random.normal(scale=gnoise,size=xalltr[:, len(binfeats):].shape)
                gnoise_test = np.random.normal(scale=gnoise,size=xte[:, len(binfeats):].shape)
            else:
                gnoise_train = np.zeros_like(xalltr[:, len(binfeats):])
                gnoise_test = np.zeros_like(xte[:, len(binfeats):])
            f1 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):]+gnoise_train, t_ph: tr1}
            f0 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):]+gnoise_train, t_ph: tr0}
            f1t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):]+gnoise_test, t_ph: tr1t}
            f0t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):]+gnoise_test, t_ph: tr0t}

            """Training"""
            if not test_only:
                for epoch in range(n_epoch):
                    avg_loss = 0.0
                    # loging
                    t0 = time.time()
                    # widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                    # pbar = ProgressBar(n_iter_per_epoch, widgets=widgets)
                    # pbar.start()
                    # train
                    np.random.shuffle(idx)
                    for j in range(n_iter_per_epoch):
                        # pbar.update(j)
                        batch = np.random.choice(idx, batch_size)
                        x_train, y_train, t_train = xtr[batch], ytr[batch], ttr[batch]
                        # if np.all(y_train==0.)or np.all(y_train==1.):
                        #     print("Y train batch has problem")
                        # if np.all(t_train==0.)or np.all(t_train==1.):
                        #     print("T train batch has problem")
                        info_dict = inference.update(feed_dict={x_ph_bin: x_train[:, 0:len(binfeats)],
                                                                x_ph_cont: x_train[:, len(binfeats):],
                                                                t_ph: t_train, y_ph: y_train})
                        if np.any(np.isnan(info_dict['loss'])):
                            print("During Processing {}, NaN Detected at Rep{}. Pass to next representaion".format(arg_info, i))
                            raise ValueError
                        avg_loss += info_dict['loss']
                    # print info_dict

                    avg_loss = avg_loss / n_iter_per_epoch
                    avg_loss = avg_loss / batch_size

                    # To check individual loss
                    # ytpostloss, x1postloss, x2postloss, zqzloss, logpvalid1 = sess.run([yt_post_loss, x1_post_loss, x2_post_loss, z_qz_loss, logp_valid],
                    #                                 feed_dict={x_ph_bin: xva[:, 0:len(binfeats)],
                    #                                             x_ph_cont: xva[:, len(binfeats):],
                    #                                             t_ph: tva, y_ph: yva})
                    # print("-"*20)
                    # print("Validation losses")
                    # for lss in [ytpostloss, x1postloss, x2postloss, zqzloss, logpvalid1]:
                    #     print(lss)
                    # print("-"*20)

                    # Print Evaluation Score
                    if epoch % args.earl == 0 or epoch == (n_epoch - 1):
                        logpvalid = sess.run(logp_valid, feed_dict={x_ph_bin: xva[:, 0:len(binfeats)],
                                                                    x_ph_cont: xva[:, len(binfeats):],
                                                                    t_ph: tva, y_ph: yva})
                        # Update and store the best validation model
                        if logpvalid >= best_logpvalid:
                            # print('Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(best_logpvalid, logpvalid))
                            saver.save(sess, save_model + '/{}-{}'.format(task, i))
                            best_logpvalid = logpvalid

                    # if epoch % args.print_every == 0:
                    #     y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=1, task=task)
                    #     if np.any(np.isnan(y0)) or np.any(np.isnan(y1)):
                    #         print("During Processing {}, NaN Detected at Rep{}. Pass to next representaion".format(arg_info, i))
                    #         raise ValueError
                    #
                    #     if task == 'ihdp':
                    #         y0, y1 = y0 * ys + ym, y1 * ys + ym
                    #     score_train = evaluator_train.calc_stats(y1, y0)
                    #     rmses_train = evaluator_train.y_errors(y0, y1)
                    #
                    #     y0t, y1t = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=1, task=task)
                    #     if task == 'ihdp':
                    #         y0t, y1t = y0t * ys + ym, y1t * ys + ym
                    #     score_test = evaluator_test.calc_stats(y1t, y0t)
                    #
                    #     print("Epoch: {}/{}, log p(x) >= {:0.3f}, ite_tr: {:0.3f}, ate_tr: {:0.3f}, score_tr: {:0.3f}, " \
                    #           "rmse_f_tr: {:0.3f}, rmse_cf_tr: {:0.3f}, ite_te: {:0.3f}, ate_te: {:0.3f}, score_te: {:0.3f}, " \
                    #           "dt: {:0.3f}".format(epoch + 1, n_epoch, avg_loss, score_train[0], score_train[1], score_train[2],
                    #                                rmses_train[0], rmses_train[1], score_test[0], score_test[1], score_test[2],
                    #                                time.time() - t0))
            """Evaluation"""
            if not load_model:
                load_model = save_model
            saver.restore(sess, load_model + '/{}-{}'.format(task, i))

            """Data elimination"""
            if drop_type != "random":
                # cpvr and fpvr computation with test data
                pred_y_0_samples, pred_y_1_samples = get_y0_y1(sess, y_post, f0t, f1t, \
                        shape=yte.shape, L=100, verbose=False, task=task, get_sample=True)

                loss_cpvr_0 = (1-tte) * np.var(pred_y_1_samples, axis=0)
                loss_cpvr_1 = tte * np.var(pred_y_0_samples, axis=0)
                loss_cpvr = (loss_cpvr_0 + loss_cpvr_1).squeeze() # (m, 1)

                loss_fpvr_0 = tte * np.var(pred_y_1_samples, axis=0)
                loss_fpvr_1 = (1-tte) * np.var(pred_y_0_samples, axis=0)
                loss_fpvr = (loss_fpvr_0 + loss_fpvr_1).squeeze() # (m, 1)

            if drop_type == "top":
                alpha = 1e-8

                # KL_t0 = 0.5 * tf.reduce_mean( tf.divide(tf.square(muq_t0 - mu_z), sigma_z) + tf.divide(tf.square(sigmaq_t0),sigma_z) + tf.log(tf.square(sigma_z) + alpha) - tf.log(tf.square(sigmaq_t0) + alpha*tf.ones_like(mu_z)) - tf.ones_like(mu_z) , axis=1)
                # KL_t1 = 0.5 * tf.reduce_mean( tf.divide(tf.square(muq_t1 - mu_z), sigma_z) + tf.divide(tf.square(sigmaq_t1),sigma_z) + tf.log(tf.square(sigma_z) + alpha) - tf.log(tf.square(sigmaq_t1) + alpha*tf.ones_like(mu_z)) - tf.ones_like(mu_z) , axis=1)

                KL_t0t = 0.5 * tf.reduce_mean( tf.divide(tf.square(muq_t0 - mu_z), sigma_z) + tf.divide(tf.square(sigmaq_t0),sigma_z) + tf.log(tf.square(sigma_z) + alpha) - tf.log(tf.square(sigmaq_t0) + alpha*tf.ones_like(mu_z)) - tf.ones_like(mu_z) , axis=1)
                KL_t1t = 0.5 * tf.reduce_mean( tf.divide(tf.square(muq_t1 - mu_z), sigma_z) + tf.divide(tf.square(sigmaq_t1),sigma_z) + tf.log(tf.square(sigma_z) + alpha) - tf.log(tf.square(sigmaq_t1) + alpha*tf.ones_like(mu_z)) - tf.ones_like(mu_z) , axis=1)


                # KL_t0 = sess.run(KL_t0, feed_dict=f0)
                # KL_t1 = sess.run(KL_t1, feed_dict=f1)
                # KL = KL_t0 + KL_t1

                KL_t0t = sess.run(KL_t0t, feed_dict=f0t)
                KL_t1t = sess.run(KL_t1t, feed_dict=f1t)
                KLt = KL_t0t + KL_t1t

                # train_top10_idx = np.argpartition(KL, -int(0.1*len(KL)))[-int(0.1*len(KL)):]
                # test_top10_idx = np.argpartition(KLt, -int(0.1*len(KLt)))[-int(0.1*len(KLt)):]
                # train_filter = [ i_filt for i_filt in range(len(KL)) if i_filt not in train_top10_idx]
                # test_filter = [ i_filt for i_filt in range(len(KLt)) if i_filt not in test_top10_idx]
                indices = np.argsort(KLt)
                # raise NotImplementedError

            elif drop_type == "random":
                # train_filter = random.sample(range(0, num_train), int(drop_ratio * num_train))
                indices = np.array(list(range(0, num_test)))
                np.random.shuffle(indices)

            elif drop_type == "cpvr":
                indices = np.argsort(loss_cpvr)

            elif drop_type == 'fpvr':
                indices = np.argsort(loss_fpvr)

            elif drop_type == 'c-f':
                indices = np.argsort(loss_cpvr - loss_fpvr)

            elif drop_type == 'f-c':
                indices = np.argsort(loss_fpvr - loss_cpvr)

            elif drop_type == 'c+f':
                indices = np.argsort(loss_fpvr + loss_cpvr)

            else:
                indices = np.array(list(range(0, num_test)))

            waste_num = int(drop_ratio * len(indices))
            test_filter = indices[:-waste_num] if waste_num != 0  else indices

            """Scoring"""
            if task == 'jobs':
                evaluator_test = Evaluator(yte[test_filter], tte[test_filter], e=ete[test_filter], task=task)
                evaluator_train = Evaluator(yalltr, talltr, e=ealltr, task=task)
            else:
                evaluator_test = Evaluator(yte[test_filter], tte[test_filter], y_cf=y_cfte[test_filter], mu0=mu0te[test_filter], mu1=mu1te[test_filter], task=task)
                evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                            mu0=np.concatenate([mu0tr, mu0va], axis=0), mu1=np.concatenate([mu1tr, mu1va], axis=0), task=task)
            y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=100, verbose=False, task=task)
            if task == 'ihdp':
                y0, y1 = y0 * ys + ym, y1 * ys + ym
            score = evaluator_train.calc_stats(y1, y0)
            scores[i, :] = score

            y0t, y1t = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=100, verbose=False, task=task)
            if task == 'ihdp':
                y0t, y1t = y0t * ys + ym, y1t * ys + ym
            score_test = evaluator_test.calc_stats(y1t[test_filter], y0t[test_filter])
            scores_test[i, :] = score_test

            # print('Replication: {}/{}, tr_ite: {:0.3f}, tr_ate: {:0.3f}, tr_score: {:0.3f}' \
            #       ', te_ite: {:0.3f}, te_ate: {:0.3f}, te_score: {:0.3f}'.format(i + 1, args.reps,
            #                                                                     score[0], score[1], score[2],
            #                                                                     score_test[0], score_test[1], score_test[2]))
            sess.close()

    print('CEVAE model total scores ' + str(arg_info))
    train_means, train_stds = np.mean(scores, axis=0), sem(scores, axis=0)
    print('train ITE: {:.3f}+-{:.3f}, train ATE: {:.3f}+-{:.3f}, train SCORE: {:.3f}+-{:.3f}' \
          ''.format(train_means[0], train_stds[0], train_means[1], train_stds[1], train_means[2], train_stds[2]))

    test_means, test_stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
    print('test ITE: {:.3f}+-{:.3f}, test ATE: {:.3f}+-{:.3f}, test SCORE: {:.3f}+-{:.3f}' \
          ''.format(test_means[0], test_stds[0], test_means[1], test_stds[1], test_means[2], test_stds[2]))

    results = (train_means[2], train_stds[2], test_means[2], test_stds[2])
    write_results(args, results)
