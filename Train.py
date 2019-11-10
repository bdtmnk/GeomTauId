
##Warning Test is required;

import time
import torch
import numpy as np
from Logger import Logger
from Model import load_model
import torch.optim as optim
from LoadModel import ECN2, ECN3
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from LoadData import TauIdDataset, get_weights
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score



TRAIN_SET = "/pnfs/desy.de/cms/tier2/store/user/ldidukh/TauId/2016/train_samples/"
TEST_SET = "/pnfs/desy.de/cms/tier2/store/user/ldidukh/TauId/2016/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/dydukhle/GeomTauID/GCN_DM_*_KNN_ECN3/"
RESUME_TRAINING = False
EPOCH = 0
RUN_TEST = True
KNN_Number = 10
TOTAL_EPOCH = 10
BATCH_SIZE = 10
N_EVENTS = 50#000



def train(config):


    start = time.time()

    # Prepare train data
    train_dataset = TauIdDataset(TRAIN_SET, num=N_EVENTS)
    train_length = train_dataset.len
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

    train = time.time()
    print("Train data loaded: {0}".format(train - start))

    # Prepare test data
    if RUN_TEST:
        test_dataset = TauIdDataset(TEST_SET,  num=N_EVENTS)
        test_length = test_dataset.len
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
        test = time.time()
        print("Test data loaded: {0}".format(test - train))



    #Start Summary Writer:
    writer = SummaryWriter()

    # Load or create the network
    if RESUME_TRAINING:
        net, optimizer, _, scheduler = load_model('{0}ECN_{1}.pt'.format(TRAINING_RES, EPOCH))
        print ("Model loaded")
    else:
        net = ECN3(config)
        optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'])
        EPOCH = 0

    #LR Scedule for the Optimiser:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    log = Logger(TRAINING_RES, 'ECN',  RESUME_TRAINING)
    torch.set_num_threads(1)

    ##Training Starts Here:
    for epoch in range(EPOCH + 1, TOTAL_EPOCH):
	
        train_labels_list=[]
        train_outputs_list=[]
        train_loss_list=[]

        for data in iter(train_loader):

            inputs, labels = data.x, data.y.type(torch.FloatTensor)

            print("Balance: ", inputs.detach().numpy()==0)[0].shape[0]/ inputs.detach().numpy()==1)[0].shape[0]
            

            # Compute the weights of events
            ##Scaled PT!;
            ##TODO Replace with another scaler;
            pt_train = inputs.detach().numpy()[:, 3]
            _, ind = np.unique(data.batch, return_index=True)
            pt_train = pt_train[ind]
            Y = labels.detach().numpy()
            weight = get_weights(pt_train, Y)

            #TOTO Get the DM and process of the event;

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data)

            #Loss function:
            loss = F.binary_cross_entropy(outputs, labels, weight=weight)
            loss.backward()
            optimizer.step()

            train_labels_list.append(labels.detach().numpy())
            train_outputs_list.append(outputs.detach().numpy())
            train_loss_list.append(F.binary_cross_entropy(outputs, labels, weight=weight))
            train_loss_uw_list.append(F.binary_cross_entropy(outputs, labels))
            log.eval_train(epoch, loss_list, labels_list, outputs_list)
            train_accuracy_list.append(accuracy_score(labels, outputs))
            

            index_tau = np.where(labels==1)[0]
            train_efficiency_list.append(accuracy_score(labels[index_labels], outputs[index_labels]))
            print("Efficiency:  ", train_efficiency_list, "Accuracy:  ",train_accuracy_list)

            if epoch % 10 ==0:
                writer.add_scalar('Loss/train', train_loss_list, n_iter)
                writer.add_scalar('uwLoss/train', train_loss_uw_list, n_iter)
                writer.add_scalar('Accuracy/train', train_accuracy_list, n_iter)
                writer.add_scalar('Efficiency/train', train_efficiency_list, n_iter)

                #TODO add Efficiency and AUC:
           

        scheduler.step()
	   
        #log.save_train(epoch)
        torch.save({
            'epoch': epoch,
            'net': net,
            'optimizer': optimizer,
            'loss': loss,
            'scheduler': scheduler
        }, '{0}ECN_{1}.pt'.format(TRAINING_RES, epoch))

	n_iter = epoch
	
        # Run validation
        if RUN_TEST and epoch % 10:0
            with torch.no_grad():
                labels_list = []
                outputs_list = []
                loss_list = []
                test_accuracy_list = []
                test_efficiency_list = []

                for data in iter(test_loader):
                    inputs, labels = data.x, data.y.type(torch.FloatTensor)
                    outputs = net(data)
                    labels_list.append(labels)
                    outputs_list.append(outputs)
                    loss_list.append(F.binary_cross_entropy(outputs, labels))
                    index_tau = np.where(np.array(labels)==1)[0]
                    test_efficiency_list.append(accuracy_score(labels[index_labels], outputs[index_labels]))
                    test_accuracy_list.append(accuracy_score(labels, outputs))
        
            log.eval_test(epoch, loss_list, labels_list, outputs_list)
            writer.add_scalar('Loss/test', loss_list, n_iter)
            writer.add_scalar('Accuracy/test', test_accuracy_list, n_iter)
            writer.add_scalar('Efficiency/test', test_efficiency_list, n_iter)
            #TOTO add Rejection rate for the backgroud:
            #TODO add the unweighted loss:
            #TODO add dm and process disctirbution:

    log.close()
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    print('Finished Training')

if __name__ == "__main__":
    train()
