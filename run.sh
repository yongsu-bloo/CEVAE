# IHDP-100 opt-params
# task    epochs  lamba   lr  batch_size  latent_dim  nh  h	train_mean	train_std	test_mean	test_std
# ihdp    2000  1e-6    3.16e-05	256	40	2	256	2.33580447997	0.319084194844	2.53072932291	0.346911865924
#
# JOBS-10 opt-params
# task	lr	epochs	lamba	batch_size	latent_dim	nh	h_dim	train_mean	train_std	test_mean	test_std
# jobs	3.16e-05	100	0.0001	512	10	4	256	0.23022240108	0.00320491608665	0.219133406779	0.0219603794075
CUDA_VISIBLE_DEVICES="1" python cevae.py --exp_name opt-param --task ihdp --lr 3.16e-05 --epochs 500 --lamba 1e-6 --batch_size 256 --latent_dim 40 --nh 2 --h_dim 256 --rep 1000 &> logs/ihdp.txt &
# CUDA_VISIBLE_DEVICES="2" python cevae.py --exp_name opt-param --task jobs --lr 3.16e-05 --epochs 100 --lamba 0.0001 --batch_size 512 --latent_dim 10 --nh 4 --h_dim 256 --rep 10 &> logs/jobs.txt &
CUDA_VISIBLE_DEVICES="3" python cevae.py --exp_name opt-param --save_model models/opt-param2 --task ihdp --lr 3.16e-05 --epochs 3000 --lamba 1e-6 --batch_size 256 --latent_dim 40 --nh 2 --h_dim 256 --rep 1000 &> logs/ihdp2.txt &
