##Warning Test is required;
import time
import torch
import numpy as np
from Logger import Logger
import torch.optim as optim
from LoadModel import ECN2, ECN3
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from LoadData import TauIdDataset, get_weights
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from LoadModel import load_model
from torchcontrib.optim import SWA


TRAIN_SET = "/pnfs/desy.de/cms/tier2/store/user/ldidukh/TauId/2016/train_samples/"
TEST_SET = "/pnfs/desy.de/cms/tier2/store/user/ldidukh/TauId/2016/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/dydukhle/GeomTauID/GCN_QCD_KNN_468_ECN3/"
RESUME_TRAINING = False
EPOCH = 0
RUN_TEST = True
TOTAL_EPOCH = 100
BATCH_SIZE = 1024
N_EVENTS = 2#50#000
N_FILES = 2#2
PROC = "QCD"#WJ all


num_workers = 16

config = {}
config['lr'] = 0.5
config['momentum'] = 0.9
config['KNN_Number'] = 4


if __name__ == "__main__":

    start = time.time()

    # Prepare train data
    train_dataset = TauIdDataset(TRAIN_SET, num=N_EVENTS, nfiles=N_FILES, processes=PROC)
    train_length = train_dataset.len
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

    train = time.time()
    print("Train data loaded: {0}".format(train - start))

    if RUN_TEST:
        test_dataset = TauIdDataset(TEST_SET,  num=N_EVENTS,  nfiles=N_FILES, processes=PROC)
        test_length = test_dataset.len
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
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
        opt = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'])
        EPOCH = 0

    #LR Scedule for the Optimiser
    #opt = SWA(optimizer)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    log = Logger(TRAINING_RES, 'ECN',  RESUME_TRAINING)
    torch.set_num_threads(1)

    ##Training Starts Here:
    for epoch in range(EPOCH + 1, TOTAL_EPOCH):

        train_labels_list=[]
        train_outputs_list=[]
        train_loss_list=[]
        train_loss_uw_list = []
        train_accuracy_list = []
        train_efficiency_list = []

        for data in iter(train_loader):

            inputs, labels = data.x, data.y.type(torch.FloatTensor)
            #print("Balance: ", inputs.detach().numpy()==0)[0].shape[0]/ inputs.detach().numpy()==1)[0].shape[0]


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
            opt.zero_grad()

            # forward + backward + optimize
            outputs = net(data)

            #Loss function:
            loss = F.binary_cross_entropy(outputs, labels, weight=weight)
            loss.backward()
            opt.step()

            train_labels_list.append(labels.detach().numpy())
            train_outputs_list.append(outputs.detach().numpy())
            train_loss_list.append(F.binary_cross_entropy(outputs, labels, weight=weight).detach().numpy())
            train_loss_uw_list.append(F.binary_cross_entropy(outputs, labels).detach().numpy())
            train_accuracy_list.append(accuracy_score(labels.detach().numpy(), outputs.detach().numpy().round()))
            index_labels = np.where(labels.detach().numpy()==1)[0]
            train_efficiency_list.append(accuracy_score(labels.detach().numpy()[index_labels], outputs.detach().numpy().round()[index_labels]))
            log.eval_train(loss, labels, outputs)

            print("Epoch: ",epoch, " Efficiency:  ", train_efficiency_list[-1], "Accuracy:  ",train_accuracy_list[-1])

            if epoch % 1 ==0:
                scheduler.step()
		#opt.update_swa()
		writer.add_scalar('Loss/train', train_loss_list[-1], epoch)
                writer.add_scalar('uwLoss/train', train_loss_uw_list[-1], epoch)
                writer.add_scalar('Accuracy/train', train_accuracy_list[-1], epoch)
                writer.add_scalar('Efficiency/train', train_efficiency_list[-1], epoch)
#		writer.add_histogram("DM/train", train_dm, bins=30, global_step=epoch)
		writer.add_image("Position/train", data.pos.detach().numpy()[1], global_step=epoch)
                #TODO add Efficiency and AUC:
	#opt.swap_swa_sgd()
	print(opt)
        #log.save_train(epoch)
        torch.save({
            'epoch': epoch,
            'net': net,
            'optimizer': opt,
            'loss': loss,
        }, '{0}ECN_{1}.pt'.format(TRAINING_RES, epoch))

        # Run validation
        if epoch % 10 == 0:
	   if RUN_TEST==False: continue
	   with torch.no_grad():
                labels_list = []
                outputs_list = []
                loss_list = []
                test_accuracy_list = []
                test_efficiency_list = []

                for data in iter(test_loader):
                    inputs, labels = data.x, data.y.type(torch.FloatTensor)
                    outputs = net(data)
                    labels_list.append(labels.detach().numpy())
                    outputs_list.append(outputs.detach().numpy())
	            loss = F.binary_cross_entropy(outputs, labels)
                    loss_list.append(F.binary_cross_entropy(outputs, labels).detach().numpy())
                    index_labels = np.where(np.array(labels.detach().numpy())==1)[0]
                    test_efficiency_list.append(accuracy_score(labels.detach().numpy()[index_labels],
                                                               outputs.detach().numpy().round()[index_labels]))
                    test_accuracy_list.append(accuracy_score(labels.detach().numpy(), outputs.detach().numpy().round())) 
#		    print("Test Epoch: ",epoch, " Efficiency:  ", test_efficiency_list[-1], "Accuracy:  ",test_accuracy_list[-1]) 
                    log.eval_test(epoch,loss, labels, outputs)
                    writer.add_scalar('Loss/test', loss_list[-1], epoch)
                    writer.add_scalar('Accuracy/test', test_accuracy_list[-1], epoch)
                    writer.add_scalar('Efficiency/test', test_efficiency_list[-1], epoch)
		    writer.add_image("Position/test", data.pos.detach().numpy()[1], global_step=epoch)
            #TOTO add Rejection rate for the backgroud:
            #TODO add the unweighted loss:
            #TODO add dm and process disctirbution:

    log.close()
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    print('Finished Training')
