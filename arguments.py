from argparse import ArgumentParser

"""
IHDP-100 opt-params
task    epochs  lamba   lr  batch_size  latent_dim  nh  h	train_mean	train_std	test_mean	test_std
ihdp    3000    1e-6    3.16e-05	256	40	2	256	2.33580447997	0.319084194844	2.53072932291	0.346911865924

JOBS-10 opt-params
task	lr	epochs	lamba	batch_size	latent_dim	nh	h_dim	train_mean	train_std	test_mean	test_std
jobs	3.16e-05	100	0.0001	512	10	4	256	0.23022240108	0.00320491608665	0.219133406779	0.0219603794075

TWINS-100 opt-params
task	lr	epochs	lamba	batch_size	latent_dim	nh	h_dim	train_mean	train_std	test_mean	test_std
twins	0.0001	100	0.0001	256	20	2	128	0.809039570646	0.0022890816649	0.795459987319	0.00325886629134
+
task	lr	epochs	lamba	batch_size	latent_dim	nh	h_dim	train_mean	train_std	test_mean	test_std
twins	3.16e-05	300	0.0001	256	20	2	128	0.850773120665	0.00147659755329	0.831056339668	0.00247766484294
"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="", help="experiment_name")
    parser.add_argument('--reps', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--opt', choices=['adam', 'adamax'], default='adam')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lamba', type=float, default=0.0001)
    parser.add_argument('--earl', type=int, default=100, help="validation interval epochs")
    parser.add_argument('--print_every', type=int, default=20)
    # added from cib research
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=20)
    parser.add_argument('--task', type=str, default='ihdp')
    parser.add_argument('--nh', type=int, default=3)
    parser.add_argument('--h_dim', type=int, default=200)

    # gaussian noise experiments
    parser.add_argument('--noise', type=int, default=0)

    # padding bernoulli noise exp
    parser.add_argument('--pnoise', type=str, default=None, help="padding noise type")
    parser.add_argument('--pn_size', type=int, default=10)
    parser.add_argument('--pn_scale', type=float, default=0.2)
    # uncertainty calibration
    parser.add_argument('--drop_type', type=str, help='type of drop the samples with high uncertainty. ["kl", "random", "cpvr", "fpvr", "c-f", "f-c", "c+f"]')
    parser.add_argument('--drop_ratio', type=float, default=0.0, help='ratio of drop the samples with high uncertainty ')
    # path
    parser.add_argument('--data_path', type=str, default="../data/IHDP/", help="data path")
    parser.add_argument('--data_pref', type=str, default="", help="data path prefix")
    parser.add_argument('--save_model', type=str, default=None, help="model path to save.")
    parser.add_argument('--load_model', type=str, default=None, help="model path to load")
    parser.add_argument('--evaluate', type=bool, default=False, help="Test only")

    parser.add_argument('--data_ratio', type=float, default=1.0)

    # adversary
    parser.add_argument('--adv_scale', type=float, default=0.0)

    args = parser.parse_args()
    args.true_post = True

    return args
