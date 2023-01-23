import argparse

def get_config(args):
    
    parser = argparse.ArgumentParser()
    
    # DATA
    
    parser.add_argument('--spacecraft', type=str, default='all', help="Which spacecraft data should be processed (all, Wind, STEREO-A, STEREO-B).")
    parser.add_argument('--split', type=int, default=1, help="Which split should be used (btw 1 and 5)")
    parser.add_argument('--resampling', type=str, default="10T", help="Which sampling rate should be used for preprocessing")
    parser.add_argument('--feature', type=str, default='ICME', help="Name of the structure to be identified")
    parser.add_argument('--stride', type=int, default=120, help="Stride used for sliding window during training")
    parser.add_argument('--test_stride', default=None, help="Stride used for sliding window during testing")
    parser.add_argument('--sampling_rate', default = 1, help='Sampling rate used for data generator')
    parser.add_argument('--catalog', choices={"helcat", "allcat","nguyen"}, default="helcat", help="Which catalog to use.") 
    parser.add_argument('--shift', default=0, help="shift to use for prediction")
    
    
    # I/O
    
    parser.add_argument('--data_dir', default='datasets/files/', help='Data directory')
    parser.add_argument('--save_dir', default='saved_models', help='Saved model directory')
    
    
    
    # TRAINING PROCESS
    
    parser.add_argument('--task', choices={"seq2seq"},    default="seq2seq", help=("Training objective/task: sequence 2 sequence classification"))
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--optimizer', choices={"Adam", "RAdam"}, default="Adam", help="Optimizer")
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--max_lr', type=float, default=0.01, help='max learning rate')
    parser.add_argument('--base_lr', type=float, default=0.00001, help='base learning rate')
    parser.add_argument('--step_size', type=float, default=1000, help='step size learning rate')
    parser.add_argument('--schedule',choices={"cyclic", "warmup"},    default="cyclic", help="Learning Rate Schedule. ")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to use for fixed lr')
    
    parser.add_argument('--earlystopping_patience', default=50, help='epochs to wait until training is terminated, set to None otherwise') 
    parser.add_argument('--shuffle', default=False, help='whether to shuffle data or not') 
    parser.add_argument('--loss', default='dice_loss', help= 'which loss to use')
    
    # MODEL
    
    parser.add_argument('--model', choices={"resunet"}, default="resunet",help="Model class")
    parser.add_argument('--data_window_len', type=int, default = 1024, help="Length of input sequence (size of layers).")
    
    # MODEL - RESUNET
    
    parser.add_argument('--n_filters_max', default= 512
        , help="Number of filters used in deepest layer")
    parser.add_argument('--res_depth', default= 2
        , help="How many residual blocks to use in encoder")
    parser.add_argument('--reduce_factor', default= 4
        , help="How many residual blocks to use in encoder")
    parser.add_argument('--class_weights', default= False
        , help="Whether to use weighting or not.")
    
    args = parser.parse_args(args)

    config = args.__dict__
    
    return config