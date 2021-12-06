import os
import os.path as path

from splicing.utils.utils import get_architecture


def get_args(parser):
    # parser.add_argument('-spl_dr', '--splice_data_root',
    #                     dest='splice_data_root', type=str)
    # parser.add_argument('-gph_dr', '--graph_data_root',
    #                     dest='graph_data_root', type=str)
    # parser.add_argument('-res_d', '--results_dir',
    #                     dest='results_dir', type=str)

    # Workflow
    parser.add_argument('-pretrain', action='store_true')
    parser.add_argument('-save_feats', action='store_true')
    parser.add_argument('-finetune', action='store_true')

    # Directory
    parser.add_argument('-modelid', type=str, default="0", dest='model_id',
        help='Model identifier, placed in front of other folder name defining args.')

    # Model loading 
    parser.add_argument('-load_pretrained', action='store_true')
    parser.add_argument(
        '-mpass', '--model_pass', type=int, default=10, dest='model_pass',
        help='The pass number of the SpliceAI model to load.')
    parser.add_argument(
        '-midx', '--model_index', type=int, default=1, dest='model_index',
        help='The index of the SpliceAI model to load.')
    parser.add_argument(
        '-mit', '--model_iteration', type=int, required=False,
        dest='model_iteration',
        help='The iteration (pass) of the SpliceAI model to load.')
    parser.add_argument('-load_gcn', action='store_true')

    # SpliceAI / Pretraining args
    parser.add_argument(
        '-cl', '--context_length', dest='context_length',
        type=int, default=400, help='The context length to use.')
    parser.add_argument(
        '-w', '--class_weights', type=int, nargs=3, default=(1, 1, 1),
        dest='class_weights', help='Class weights to use.')
    parser.add_argument(
        '-trat', '--train_ratio', type=float, default=0.9,
        dest='train_ratio', help='The proportion of the train data to use.')
    parser.add_argument(
        '-npass', '--passes', type=int, default=10,
        dest='passes', help='Number of passes over the train dataset to do.')

    
    ### Finetuning / Graph Training Args
    # Graph initialization
    parser.add_argument(
        '-nc', '--n_channels', type=int, default=32, dest='n_channels',
        help='Number of convolution channels to use.'
    )
    parser.add_argument(
        '-nhidd', '--hidden_size', type=int, default=128, dest='hidden_size',
        help='The dimensionality of the hidden layer in the graph network.'
    )
    parser.add_argument('-gcn_dropout', type=float, default=0.2)

    parser.add_argument('-gcn_layers', type=int, default=2)
    parser.add_argument('-A_saliency', action='store_true')
    parser.add_argument('-adj_type', type=str,
        choices=['constant', 'hic', 'both', 'random', 'none', ''], 
        default='hic'
    )
    parser.add_argument('-hicnorm', type=str,
                        choices=['KR', 'VC', 'SQRTVC', ''], default='SQRTVC')
    parser.add_argument('-hicsize', type=str,
                        choices=['125000', '250000', '500000', '1000000'],
                        default='500000')
    parser.add_argument('-gate', action='store_true')
    parser.add_argument(
        '-gbs', '--graph_batch_size', dest='graph_batch_size', type=int,
        default=1024, help='Batch size for finetuning.')
    parser.add_argument(
        '-fep', '--finetune_epochs', dest='finetune_epochs', type=int,
        default=4, help='Number of epochs for graph training.')
    
    parser.add_argument(
        '-rep', '--node_representation', 
        type=str, 
        default='average',
        dest='node_representation',
        choices = ['average', 'max', 'min', 'min-max', 'Conv1d'],
        help='How to construct the node representation '
             'from the nucleotide representations.'
    )
    parser.add_argument('-optim', type=str, choices=['adam', 'sgd'],
        default='adam')
    

    # Training args applicable to both -pretrain and -finetune
    parser.add_argument('-lr', type=float, default=0.001)
    #parser.add_argument('-lr_step_size', type=int, default=1)
    parser.add_argument(
        '-vi', '--validation_interval', type=int, default=32,
        dest='validation_interval', help='Per how many epochs to validate.')
    parser.add_argument(
        '-li', '--log_interval', type=int, default=32,
        dest='log_interval', help='Per how many updates to log to WandB.')

    # Logging Args
    parser.add_argument('-wb', '--wandb', dest='wandb', action='store_true')

    # need to assign these:
    parser.add_argument('-test_batch_size', type=int, default=512)
    parser.add_argument('-weight_decay', type=float, default=5e-5,
                        help='weight decay')
    parser.add_argument('-lr_decay', type=float, default=0)
    
    parser.add_argument('-lr_decay2', type=float, default=0) # triggers update of scheduler at every epoch if > 1
    parser.add_argument('-lr_step_size2', type=int, default=100)
    parser.add_argument('-dropout', type=float, default=0.1)
    
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'],
                        default='best')
    parser.add_argument('-br_threshold', type=float, default=0.5)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-shuffle_train', action='store_true')
    parser.add_argument('-viz', action='store_true')
    parser.add_argument('-gpu_id', type=int, default=-1)
    parser.add_argument('-small', action='store_true')
    parser.add_argument('-summarize_data', action='store_true')
    parser.add_argument('-overwrite', action='store_true')
    parser.add_argument('-test_only', action='store_true')
    parser.add_argument('-seq_length', type=int, default=2000)

    # would remove these and hard code default values:
    parser.add_argument('-cell_type', type=str, default='GM12878')
    parser.add_argument('-lr2', type=float, default=0.002)
    
    opt = parser.parse_args()
    return opt


def config_args(opt, config):

    # OUR DATA DIRECTORIES
    opt.splice_data_root = path.join(
        config['DATA_DIRECTORY'], config['DATA_PIPELINE']['output_dir'])
    opt.results_dir = path.join(
        config['DATA_DIRECTORY'], config['TRAINING']['results_dir'])

    # Workflow
    assert (sum([opt.pretrain, opt.save_feats, opt.finetune]) == 1, 
        "Must have only one of: -pretrain, -save_feats, -finetune")
    
    if opt.pretrain:
        opt.workflow = "pretrain"
    elif opt.save_feats:
        opt.workflow = "save_feats"
    else:
        opt.workflow = "finetune"

    opt.dec_dropout = opt.dropout

    opt.model_name = f'{opt.model_id}_graph.splice_ai'
    opt.model_name += '.' + str(opt.optim)
    opt.model_name += '.lr_' + str(opt.lr).split('.')[1]

    #if opt.lr_decay > 0:
    #    opt.model_name += '.decay_' + str(opt.lr_decay).replace(
    #        '.', '') + '_' + str(opt.lr_step_size)

    opt.model_name += '.drop_' + ("%.2f" % opt.dropout).split('.')[1] + '_' + \
                      ("%.2f" % opt.dec_dropout).split('.')[1]

    if opt.save_feats:
        #opt.model_name += '.save_feats'
        opt.pretrain = False
        opt.shuffle_train = False
        opt.train_ratio = 1.0
        # opt.epochs = 1

    if opt.finetune:
        opt.model_name += '/finetune'
        opt.model_name += '.lr2_' + str(opt.lr2).split('.')[1]
        opt.model_name += '.gcndrop_' + (
                "%.2f" % opt.gcn_dropout).split('.')[1]
        opt.model_name += '.' + str(opt.optim2)
        opt.model_name += '.layers_' + str(opt.gcn_layers)
        if opt.gate:
            opt.model_name += '.gate'
        opt.model_name += '.adj_' + opt.adj_type
        if opt.adj_type == 'hic' or opt.adj_type == 'both':
            opt.model_name += '.norm_' + opt.hicnorm
        if opt.lr_decay > 0:
            opt.model_name += '.decay_' + str(opt.lr_decay).replace(
                '.', '') + '_' + str(opt.lr_step_size)

    opt.model_name = path.join(opt.results_dir, opt.cell_type, opt.model_name)

    # TODO
    opt.graph_data_root = path.join(
        config['DATA_DIRECTORY'], config['DATA_PIPELINE']['output_dir'])
    opt.dataset = path.join(opt.graph_data_root, opt.cell_type)
    opt.cuda = not opt.no_cuda

    if opt.load_gcn:
        opt.model_name += '.load_gcn'

    # if (not opt.viz) \
    #         and (not opt.overwrite) \
    #         and ('test' not in opt.model_name) \
    #         and (path.exists(opt.model_name)) \
    #         and (not opt.load_gcn) \
    #         and (not opt.save_feats):
    #     print(opt.model_name)
    #     overwrite_status = input('Already Exists. Overwrite?: ')
    #     if overwrite_status == 'rm':
    #         os.system('rm -rf ' + opt.model_name)
    #     elif 'y' not in overwrite_status.lower():
    #         exit(0)

    # CNN args:
    kernel_size, dilation_rate, batch_size = get_architecture(
        opt.context_length)

    opt.kernel_size = kernel_size
    opt.dilation_rate = dilation_rate
    opt.batch_size = batch_size

    opt.window_size = config['DATA_PIPELINE']['window_size']

    if not opt.pretrain:
        opt.batch_size = 512
        opt.test_batch_size = 512

    # Graph args:
    if opt.node_representation == "min-max":
        opt.n_channels = opt.n_channels * 2

    # Directory handling
    assert not (os.path.exists(opt.model_name) and opt.pretrain), \
        "Model directory already exists, please specify another -modelid"

    assert not (os.path.exists(opt.model_name) and opt.finetune), \
        "Finetune model directory already exists"

    return opt
