import os
import argparse

# python eval_50.py --chkpt_dir C:\Users\eunic\Desktop\Github\NN\project\outputs\checkpoints\train_250\checkpoints --name train_250_eval

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
