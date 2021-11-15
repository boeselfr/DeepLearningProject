import logging
import os
from os import path

import torch

from splicing.utils import metrics
from splicing.utils.graph_utils import directory_setup

FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(__name__)


# TODO

def compute_metrics(all_predictions,all_targets,loss,args,elapsed,data_dict,cell_type,per_label_type=False,plot=False,model_name=None):
    metrics_dict = {}
    if per_label_type:
        label_list = [key.replace('wgencodeawg','').replace('unipk','').replace('gm12878','').replace('k562','').replace('iggmus','').replace('syd','').replace('uta','').replace('haib','').replace('pcr1x','').replace('pcr2x','').replace('tfbs','tfbs_').replace('iggrab','').replace('broad','').split('sc')[0] for key in data_dict['tgt'].keys()]

        tfbs_indices = [i for i,label in enumerate(label_list) if 'tfbs' in label]
        if cell_type == 'GM12878':
            hm_indices = [i for i,label in enumerate(label_list) if 'e116-h' in label] # GM12878
        else:
            hm_indices = [i for i,label in enumerate(label_list) if 'e123-h' in label] # K562
        dnase_indices = [i for i,label in enumerate(label_list) if 'dnase' in label]

        tfbs_preds = torch.index_select(all_predictions,1,torch.Tensor(tfbs_indices).long()).numpy()
        tfbs_targets = torch.index_select(all_targets,1,torch.Tensor(tfbs_indices).long()).numpy()
        hm_preds = torch.index_select(all_predictions,1,torch.Tensor(hm_indices).long()).numpy()
        hm_targets = torch.index_select(all_targets,1,torch.Tensor(hm_indices).long()).numpy()
        dnase_preds = torch.index_select(all_predictions,1,torch.Tensor(dnase_indices).long()).numpy()
        dnase_targets = torch.index_select(all_targets,1,torch.Tensor(dnase_indices).long()).numpy()

        

        tfbs_meanAUC,medianAUC,varAUC,allAUC = metrics.auroc(tfbs_targets, tfbs_preds)
        tfbs_meanAUPR,medianAUPR,varAUPR,allAUPR = metrics.aupr(tfbs_targets, tfbs_preds)
        tfbs_meanFDR,medianFDR,varFDR,allFDR = metrics.fdr(tfbs_targets, tfbs_preds)
        hm_meanAUC,medianAUC,varAUC,allAUC = metrics.auroc(hm_targets, hm_preds)
        hm_meanAUPR,medianAUPR,varAUPRz,allAUPR = metrics.aupr(hm_targets, hm_preds)
        hm_meanFDR,medianFDR,varFDR,allFDR = metrics.fdr(hm_targets, hm_preds)
        dnase_meanAUC,medianAUC,varAUC,allAUC = metrics.auroc(dnase_targets, dnase_preds)
        dnase_meanAUPR,medianAUPR,varAUPR,allAUPR = metrics.aupr(dnase_targets, dnase_preds)
        dnase_meanFDR,medianFDR,varFDR,allFDR = metrics.fdr(dnase_targets, dnase_preds)

        metrics_dict['tfbs_meanAUC'] = tfbs_meanAUC
        metrics_dict['tfbs_meanAUPR'] = tfbs_meanAUPR
        metrics_dict['tfbs_meanFDR'] = tfbs_meanFDR

        metrics_dict['hm_meanAUC'] = hm_meanAUC
        metrics_dict['hm_meanAUPR'] = hm_meanAUPR
        metrics_dict['hm_meanFDR'] = hm_meanFDR

        metrics_dict['dnase_meanAUC'] = dnase_meanAUC
        metrics_dict['dnase_meanAUPR'] = dnase_meanAUPR
        metrics_dict['dnase_meanFDR'] = dnase_meanFDR


        if plot:
            if 'gcn' in model_name:
                model_name = 'ChromeGCN$_{HiC}$'
                save_name = 'ChromeGCN'
            else:
                model_name = 'CNN'
                save_name = 'CNN'

            print('plotting')
            metrics.plot_auroc(tfbs_targets, tfbs_preds, 'TF (' + model_name + ')', cell_type + '_' + save_name + '_TF')
            metrics.plot_auroc(hm_targets, hm_preds, 'HM (' + model_name + ')', cell_type + '_' + save_name + '_HM')
            metrics.plot_auroc(dnase_targets, dnase_preds, 'DNase I (' + model_name + ')', cell_type + '_' + save_name + '_DNase')
            metrics.plot_aupr(tfbs_targets, tfbs_preds, 'TF (' + model_name + ')', cell_type + '_' + save_name + '_TF')
            metrics.plot_aupr(hm_targets, hm_preds, 'HM (' + model_name + ')', cell_type + '_' + save_name + '_HM')
            metrics.plot_aupr(dnase_targets, dnase_preds, 'DNase I (' + model_name + ')', cell_type + '_' + save_name + '_DNase')

        all_targets = all_targets.numpy()
        all_predictions = all_predictions.numpy()
        
    meanAUC,medianAUC,varAUC,allAUC = metrics.auroc(all_targets, all_predictions)
    meanAUPR,medianAUPR,varAUPR,allAUPR = metrics.aupr(all_targets, all_predictions)
    meanFDR,medianFDR,varFDR,allFDR = metrics.fdr(all_targets, all_predictions)
    mAP = metrics.mean_average_precision(all_targets, all_predictions)

    optimal_threshold = args.br_threshold
    
    # optimal_thresholds = Find_Optimal_Cutoff(all_targets,all_predictions)
    # optimal_threshold = numpy.mean(numpy.array(optimal_thresholds))

    all_predictions[all_predictions < optimal_threshold] = 0
    all_predictions[all_predictions >= optimal_threshold] = 1

    print('mAP:      '+str(round(mAP,3)))
    print('meanAUC:  '+str(round(meanAUC,3)))
    print('meanAUPR: '+str(round(meanAUPR,3)))
    print('meanFDR:  '+str(round(meanFDR,3)))

    metrics_dict['mAP'] = mAP
    metrics_dict['meanAUC'] = meanAUC
    metrics_dict['medianAUC'] = medianAUC
    metrics_dict['allAUC'] = allAUC
    metrics_dict['allFDR'] = allFDR
    metrics_dict['meanAUPR'] = meanAUPR
    metrics_dict['medianAUPR'] = medianAUPR
    metrics_dict['allAUPR'] = allAUPR
    metrics_dict['meanFDR'] = meanFDR
    metrics_dict['medianFDR'] = medianFDR
    metrics_dict['loss'] = loss
    metrics_dict['time'] = elapsed

    return metrics_dict

class Logger:
    def __init__(self,args):
        self.model_name = args.model_name

        if args.model_name:
            try:
                os.makedirs(args.model_name)
            except OSError as exc:
                pass

            try:
                os.makedirs(args.model_name+'/epochs/')
            except OSError as exc:
                pass

            self.write_output_files = False

            if self.write_output_files:
                self.file_names = {}
                self.file_names['train'] = os.path.join(args.model_name,'train_results.csv')
                self.file_names['valid'] = os.path.join(args.model_name,'valid_results.csv')
                self.file_names['test'] = os.path.join(args.model_name,'test_results.csv')

                self.file_names['valid_all_aupr'] = os.path.join(args.model_name,'valid_all_aupr.csv')
                self.file_names['valid_all_auc'] = os.path.join(args.model_name,'valid_all_auc.csv')
                self.file_names['test_all_aupr'] = os.path.join(args.model_name,'test_all_aupr.csv')
                self.file_names['test_all_auc'] = os.path.join(args.model_name,'test_all_auc.csv')
                

                f = open(self.file_names['train'],'w+'); f.close()
                f = open(self.file_names['valid'],'w+'); f.close()
                f = open(self.file_names['test'],'w+'); f.close()
                f = open(self.file_names['valid_all_aupr'],'w+'); f.close()
                f = open(self.file_names['valid_all_auc'],'w+'); f.close()
                f = open(self.file_names['test_all_aupr'],'w+'); f.close()
                f = open(self.file_names['test_all_auc'],'w+'); f.close()
            os.utime(args.model_name,None)
        
        self.best_valid = {'loss':1000000,'ACC':0,'HA':0,'ebF1':0,'miF1':0,'maF1':0,'meanAUC':0,'medianAUC':0,'meanAUPR':0,'medianAUPR':0,'meanFDR':0,'medianFDR':0,'allAUC':None,'allAUPR':None}

        self.best_test = {'loss':1000000,'ACC':0,'HA':0,'ebF1':0,'miF1':0,'maF1':0,'meanAUC':0,'medianAUC':0,'meanAUPR':0,'medianAUPR':0,'meanFDR':0,'medianFDR':0,'allAUC':None,'allAUPR':None,'epoch':0}


    def evaluate(self,train_metrics,valid_metrics,test_metrics,epoch,num_params,verbose=False):
        if self.model_name:
            if self.write_output_files:
                if train_metrics is not None:
                    with open(self.file_names['train'],'a') as f:
                        f.write(str(epoch)+','+str(train_metrics['loss'])
                                        +','+str(train_metrics['meanAUC'])
                                        +','+str(train_metrics['meanAUPR'])
                                        +','+str(train_metrics['meanFDR'])
                                        +','+'{elapse:3.3f}'.format(elapse=train_metrics['time'])
                                        +','+str(num_params)
                                        +'\n')
                if valid_metrics is not None:
                    with open(self.file_names['valid'],'a') as f:
                        f.write(str(epoch)+','+str(valid_metrics['loss'])
                                        +','+str(valid_metrics['meanAUC'])
                                        +','+str(valid_metrics['meanAUPR'])
                                        +','+str(valid_metrics['meanFDR'])
                                        +','+'{elapse:3.3f}'.format(elapse=train_metrics['time'])
                                        +','+'{elapse:3.3f}'.format(elapse=valid_metrics['time'])
                                        +','+str(num_params)
                                        +'\n')
                
                if test_metrics is not None:
                    with open(self.file_names['test'],'a') as f:
                        f.write(str(epoch)+','+str(test_metrics['loss'])
                                        +','+str(test_metrics['meanAUC'])
                                        +','+str(test_metrics['meanAUPR'])
                                        +','+str(test_metrics['meanFDR'])
                                        +','+'{elapse:3.3f}'.format(elapse=train_metrics['time'])
                                        +','+'{elapse:3.3f}'.format(elapse=test_metrics['time'])
                                        +','+str(num_params)
                                        +'\n')


                with open(self.file_names['valid_all_auc'],'a') as f:
                    f.write(str(epoch))
                    for i,val in enumerate(valid_metrics['allAUC']):
                        f.write(','+str(val))
                    f.write('\n')
                    f.close()

                with open(self.file_names['valid_all_aupr'],'a') as f:
                    f.write(str(epoch))
                    for i,val in enumerate(valid_metrics['allAUPR']):
                        f.write(','+str(val))
                    f.write('\n')
                    f.close()

                with open(self.file_names['test_all_auc'],'a') as f:
                    f.write(str(epoch))
                    for i,val in enumerate(test_metrics['allAUC']):
                        f.write(','+str(val))
                    f.write('\n')
                    f.close()

                with open(self.file_names['test_all_aupr'],'a') as f:
                    f.write(str(epoch))
                    for i,val in enumerate(test_metrics['allAUPR']):
                        f.write(','+str(val))
                    f.write('\n')
                    f.close()

        # stop()
        if valid_metrics is None:
            valid_metrics = test_metrics
        for metric in valid_metrics.keys():
            if not 'all' in metric and not 'time'in metric and metric in self.best_valid.keys():
                if  valid_metrics[metric] >= self.best_valid[metric]:
                    self.best_valid[metric]= valid_metrics[metric]
                    self.best_test[metric]= test_metrics[metric]
                    if metric == 'ACC':
                        self.best_test['epoch'] = epoch

         
        print('\n')
        print('**********************************')
        print('best meanAUC:  '+str(round(self.best_test['meanAUC'],4)))
        print('best meanAUPR: '+str(round(self.best_test['meanAUPR'],4)))
        print('best meanFDR: '+str(round(self.best_test['meanFDR'],4)))
        print('**********************************')

        return self.best_valid,self.best_test


def save_model(opt, epoch, model):

    model_suffix = f'SpliceAI' \
                   f'_e{epoch}' \
                   f'_cl{opt.context_length}' \
                   f'_g{opt.model_index}.h5'

    checkpoint = {'model': model.state_dict(), 'settings': opt, 'epoch': epoch}
    torch.save(checkpoint, path.join(opt.model_name, model_suffix))


class SaveLogger:
    def __init__(self, model_name):
        self.best_valid_loss = float('inf')
        self.best_valid_metric = 0
        self.best_loss_epoch = 0
        self.model_name = model_name
        self.logs_dir = path.join(model_name, 'logs')
        self.epochs_dir = path.join(self.model_name, 'epochs')
        directory_setup(self.logs_dir)
        directory_setup(self.epochs_dir)
        open(path.join(self.logs_dir, 'train.log'), 'w').close()
        open(path.join(self.logs_dir, 'valid.log'), 'w').close()
        open(path.join(self.logs_dir, 'test.log'), 'w').close()

    def save(self, epoch, opt, base_model, graph_model, valid_loss,
             valid_preds, valid_targs, test_preds=None, test_targs=None):
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.best_loss_epoch = epoch
            torch.save(valid_preds, os.path.join(
                self.epochs_dir, 'best_valid_preds_loss.pt'))
            torch.save(valid_targs, os.path.join(
                self.epochs_dir, 'best_valid_targets_loss.pt'))
            if test_preds is not None:
                torch.save(test_preds, os.path.join(
                    self.epochs_dir, 'best_test_preds_loss.pt'))
                torch.save(test_targs, os.path.join(
                    self.epochs_dir, 'best_test_targets_loss.pt'))

        # if valid_metrics_sum > self.best_valid_metric:
        #     self.best_valid_metric = valid_metrics_sum
        #     torch.save(valid_preds, os.path.join(
        #         self.epochs_dir, 'best_valid_preds_metrics.pt'))
        #     torch.save(valid_targs, os.path.join(
        #         self.epochs_dir, 'best_valid_targets_metrics.pt'))
        #     if test_preds is not None:
        #         torch.save(test_preds, os.path.join(
        #             self.epochs_dir, 'best_test_preds_metrics.pt'))
        #         torch.save(test_targs, os.path.join(
        #             self.epochs_dir, 'best_test_targets_metrics.pt'))

        if 'test' not in self.model_name and not opt.test_only:
            if opt.pretrain:
                save_model(opt, epoch, base_model)
            else:
                save_model(opt, epoch, graph_model)

    def log(self, file_name, epoch, loss, metrics):
        log_file = open(path.join(self.model_name, file_name), 'a')
        log_file.write(str(epoch) + ',' + str(loss) + ',' + str(
            metrics['mAP']) + ',' + str(metrics['meanAUC']) + ',' + str(
            metrics['meanAUPR']) + ',' + str(metrics['meanFDR']) + '\n')
        log_file.close()
