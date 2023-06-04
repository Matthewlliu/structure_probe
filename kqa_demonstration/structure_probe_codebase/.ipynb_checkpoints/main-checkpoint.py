import os
import argparse
import logging
import time

from utils import seed_everything
from generate import generate_from_local_model, generate_from_online
from model import get_model_from_local, get_model_api
from data import get_Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore")

LOCAL_MODEL = [
    'gpt2',
    'gpt2-large',
    'gpt2-XL',
    'gpt-j'
]

ONLINE_MODEL = [
    'text-davinci-001',
    'text-davinci-003',
    'chatgpt',
    'glm-130b'
]

def generate(args):
    dataset = get_Dataset(args)
    logging.info("Loading dataset completed.")
    logging.info("Dataset size: %s" % len(dataset.data))
    
    if args.model_name in LOCAL_MODEL:
        model = get_model_from_local(args)
        logging.info("Generate from local model:")
        generate_from_local_model(model, dataset, args)
    else:
        post_api = get_model_api(args)
        logging.info("Generate from online api:")
        generate_from_online(post_api, dataset, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_id', type=int,
                        default=0)
    parser.add_argument('--augment_size', type=int,
                        default=10000)
    parser.add_argument('--save_step', type=int,
                        default=2000)
    parser.add_argument('--model_name', type=str,
                        default='glm-130b')
    parser.add_argument('--model_dir', type=str,
                        default='/data/MODELS/gpt2-large')
    parser.add_argument('--logic_forms', type=str,
                        default='kopl')
    parser.add_argument('--data_dir', type=str,
                        required=True)
    parser.add_argument('--cache_dir', type=str,
                        required=True)
    parser.add_argument('--output_dir', type=str, default='',
                       help='where to store generated new data')
    parser.add_argument('--log_dir', type=str, default='',
                       help='where to store logs')
    parser.add_argument('--spare_keys', type=str,
                        default=None)
    
    parser.add_argument('--demo_num', type=int,
                       default=3)
    parser.add_argument('--batch_size', type=int,
                       default=4)
    parser.add_argument('--topk', type=int,
                       default=50)
    parser.add_argument('--topp', type=float,
                       default=0.9)
    parser.add_argument('--beam_size', type=int,
                       default=5)
    parser.add_argument('--temperature', type=float,
                       default=1)
    parser.add_argument('--strategy', type=str,
                       default='BeamSearchStrategy')
    parser.add_argument('--if_lf2nl', action='store_true')
    parser.add_argument('--toy', action='store_true')
    
    args = parser.parse_args()
    
    
    mode = 'lf2nl' if args.if_lf2nl else 'nl2lf'
    time_label = time.strftime("%Y-%m-%d", time.localtime())
    save_folder_name = '_'.join([mode, args.logic_forms, args.model_name, time_label])
    
    if args.toy:
        args.augment_size = 100
        args.save_step = 100
        save_folder_name += '_toy'
    else:
        save_folder_name += '_%s' % args.augment_size
        
    if len(args.output_dir)==0:
        args.output_dir = os.path.join(args.data_dir, save_folder_name)
    else:
        args.output_dir = args.output_dir.format(save_folder_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    if len(args.log_dir)==0:
        args.log_dir = os.path.join(args.output_dir, 'logs')
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.log_dir, '{}.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))
    
    seed_everything(501)
    
    generate(args)
    

if __name__=='__main__':
    main()