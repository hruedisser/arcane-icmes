import argparse

"options for arcane - this is used when called from a jupyter notebook"

def get_config(args):
    
    parser = argparse.ArgumentParser()
    
# DATA
    
    parser.add_argument('--spacecraft', type=str, default='Wind', help="Which spacecraft data should be processed (all, Wind, STEREO-A, STEREO-B, DSCVR, Wind_Archive).")
    parser.add_argument('--split', default=1, help="Which split should be used (btw 1 and 5) or custom")
    parser.add_argument('--splitrule', default = [[2020,2021],[2017, 2018, 2019],[2007,2008,2009,2010,2011,2012,2013,2014,2015,2016]])
    parser.add_argument('--resampling', type=str, default="10T", help="Which sampling rate should be used for preprocessing")
    parser.add_argument('--feature', type=str, default='ICME', help="Name of the structure to be identified")
    parser.add_argument('--stride', type=int, default=120, help="Stride used for sliding window during training")
    parser.add_argument('--test_stride', default=None, help="Stride used for sliding window during testing")
    parser.add_argument('--sampling_rate', default = 1, help='Sampling rate used for data generator')
    parser.add_argument('--catalog', choices={"helcat", "allcat","nguyen"}, default="helcat", help="Which catalog to use.") 
    parser.add_argument('--shift', default=0, help="shift to use for prediction")

    #FEATURES

    parser.add_argument('--std_features', default={'bx': ['bx','Bx','bx_gsm'], 'by': ['by','By','by_gsm'], 'bz': ['bz','Bz','bz_gsm'], 'bt': ['bt','Bt','B','b'], 'np': ['np','Np','proton_density', 'dens'], 'vt': ['vt','Vt','V','speed', 'proton_speed']}, help="Features to include in the Dataset")
    parser.add_argument('--add_features', default={'tp': ['tp', 'proton_temperature','Tp'],'beta': ['beta'], 'pdyn': ['pdyn'],'texrat': ['texrat']}, help="features to additionally include")


    # GENERAL

    parser.add_argument('--data_dir', default='datasets/files/', help='Data directory')
    #parser.add_argument('--save_dir', default='saved_models', help='Saved model directory')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    parser.add_argument('--GPU', default = "", help='Which GPU to use')
    parser.add_argument('--experiment_name', default = "default", help="Name of the experiment")
    parser.add_argument('--load_experiment', default = None, help = 'Name of the experiment you want to load a model from ')
    parser.add_argument('--train', default = False, help="Train the model")
    parser.add_argument('--test', default = False, help="Test the model")
    parser.add_argument('--finetune', default = False, help="Finetune the model") 
    parser.add_argument('--switch_output', default = False, help="Make classifier out of the model") 

    parser.add_argument('--realtime', default = False, help="Set to true for additional realtime testing.") 

    parser.add_argument('--seed', default = None, help="Random seed")



    # TRAINING PROCESS

    parser.add_argument('--task', choices={"seq2seq"},    default="seq2seq", help=("Training objective/task: sequence 2 sequence classification"))
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--optimizer', choices={"Adam", "RAdam"}, default="Adam", help="Optimizer")
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--max_lr', type=float, default=0.01, help='max learning rate')
    parser.add_argument('--base_lr', type=float, default=0.00001, help='base learning rate')
    parser.add_argument('--step_size', type=float, default=1000, help='step size learning rate')
    parser.add_argument('--schedule',choices={"cyclic", "warmup","simple", "custom_schedule"},    default="cyclic", help="Learning Rate Schedule. ")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to use for fixed lr')

    parser.add_argument('--earlystopping_patience', default=50, help='epochs to wait until training is terminated, set to None otherwise') 
    parser.add_argument('--reduce_lr', default=None, help='epochs to wait until training is terminated, set to None otherwise') 
    parser.add_argument('--shuffle', default=True, help='whether to shuffle data or not') 
    parser.add_argument('--loss', default='dice_loss', help= 'which loss to use')

    # MODEL

    parser.add_argument('--model', choices={"resunet"}, default="resunet",help="Model class")
    parser.add_argument('--data_window_len', type=int, default = 1024, help="Length of input sequence (size of layers).")

    # MODEL - RESUNET

    parser.add_argument('--n_filters_max', type=int, default= 512, help="Number of filters used in deepest layer")
    parser.add_argument('--res_depth', type=int, default= 2, help="How many residual blocks to use in encoder")
    parser.add_argument('--reduce_factor', type=int, default= 4, help="How many residual blocks to use in encoder")
    parser.add_argument('--class_weights', default= False, help="Whether to use weighting or not.")

    # POSTPROCESS

    parser.add_argument('--creepy', default= 3, help="Minimum duration for ICME")
    
    args = parser.parse_args(args)

    config = args.__dict__
    
    return config