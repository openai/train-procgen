import os
import argparse

# python eval_50.py --chkpt_dir C:\Users\eunic\Desktop\Github\NN\project\outputs\checkpoints\train_100\checkpoints --name train_100_eval

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--chkpt_dir', type=str, required=True)
	parser.add_argument('--name', type=str, required=True)

	args = parser.parse_args()

	command = ''
	save_dir = "./outputs/eval/%s" % (args.name)
	for chkpts in os.listdir(args.chkpt_dir):
		if chkpts.split('/')[0].isnumeric():
			model_dir = args.chkpt_dir.replace('\\', '/') + "/" + chkpts
			command += "python regularized-train/train_procgen/evaluate.py --distribution_mode easy --log_dir %s --load_model %s \n" % (save_dir + "/" + chkpts, model_dir)
	print(command)
	# parser = argparse.ArgumentParser(description='Process procgen evaluation arguments.')
	# parser.add_argument('--chkpt_dir', type=str, required=True)
	# parser.add_argument('--log_dir', type=str, default='./logs/eval')
	# parser.add_argument('--env_name', type=str, default='fruitbot')
	# parser.add_argument('--num_envs', type=int, default=64)
	# parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
	# parser.add_argument('--num_levels', type=int, default=500)
	# parser.add_argument('--start_level', type=int, default=500)
	#
	# args = parser.parse_args()
	#
	# command = "\""
	# chkpt_dir = "./outputs/log/train_50/checkpoints"
	# for chkpts in os.listdir(chkpt_dir):
	# 	model_dir = chkpt_dir + "/" + chkpts
	# 	command += "python -m train_procgen.evaluate --distribution_mode easy --log_dir ./log/val_50/%d --load_model %s &" % (chkpts, model_dir)
