import numpy as np
from LoadData import TauIdDataset, get_weights
import torch
from torch_geometric.data import Data, DataLoader
from LoadModel import ECN
import torch.nn.functional as F
import torch.optim as optim
from Logger import  Logger

# 44419 parameters
# DPF: 8838945 parameters
TRAIN_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/train_samples/"
TEST_SET = "/nfs/dust/cms/user/dydukhle/TauIdSamples/TauId/2016/test_samples/"
TRAINING_RES = "/nfs/dust/cms/user/bukinkir/TauId/ECN/"
RESUME_TRAINING = False
EPOCH = 0


def load_model(PATH):
    checkpoint = torch.load(PATH)
    net = checkpoint['net']
    optimizer = checkpoint['optimizer']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return net, optimizer, epoch, loss


if __name__ == "__main__":
    # Prepare train data
    train_dataset = TauIdDataset(TRAIN_SET, num=4096)
    train_length = train_dataset.len
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=5)

    print(train_length)

    # Prepare test data
    test_dataset = TauIdDataset(TEST_SET, num=512)
    test_length = test_dataset.len
    test_loader = DataLoader(test_dataset, batch_size=test_length, shuffle=True, num_workers=5)

    # Load or create the network
    if RESUME_TRAINING:
        net, optimizer, _, _ = load_model('{0}ECN_{1}.pt'.format(TRAINING_RES, EPOCH))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = 0.01
        print ("Model loaded")
    else:
        net = ECN()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
        EPOCH = 0

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # net = nn.DataParallel(net)

    i = 1
    # if RESUME_TRAINING:
    #     log = open(TRAINING_RES + "train.log", 'a')
    # else:
    #     log = open(TRAINING_RES + "train.log", 'w')
    # start_time = time.time()

    log = Logger(TRAINING_RES, 'ECN',  RESUME_TRAINING)

    for epoch in range(EPOCH + 1, 101):
        # Loss = acc = auc = prec = rec = 0
        j = k = 0
        for data in iter(train_loader):
            # df = pd.DataFrame()
            # for k in range(len(FEATURES)):
            #     df[FEATURES[k]] = pd.Series(df_vars.detach().numpy()[:, k])
            # df.to_csv(
            #     "{0}hists/data_e{1}b{2}.csv".format(TRAINING_RES, epoch, j),
            #     index=False)

            inputs, labels = data.x, data.y.type(torch.FloatTensor)

            # Compute the weights of events
            pt_train = inputs.detach().numpy()[:, 3]
            _, ind = np.unique(data.batch, return_index=True)
            pt_train = pt_train[ind]
            Y = labels.detach().numpy()
            weight = get_weights(pt_train, Y)

            print("Input shape", inputs.shape)
            print("Labels shape", labels.shape)

            # zero the parameter gradients
            optimizer.zero_grad()
            print("Num of features:", data.num_features)

            # forward + backward + optimize
            outputs = net(data)
            print("Output shape", outputs.shape)
            # loss = F.nll_loss(outputs, labels, weight=weight)
            loss = F.binary_cross_entropy(outputs, labels, weight=weight)
            loss.backward()
            optimizer.step()
            print(loss)

            log.eval_train(loss, labels, outputs)
            i += 1
            j += 1
        scheduler.step()

        len = j
        log.save_train(epoch)
        torch.save({
            'epoch': epoch,
            'net': net,
            'optimizer': optimizer,
            'loss': loss,
        }, '{0}ECN_{1}.pt'.format(TRAINING_RES, epoch))

        if epoch % 10 == 0:
            with torch.no_grad():
                labels_list = []
                outputs_list = []
                loss_list = []
                for data in iter(test_loader):
                    inputs, labels = data.x, data.y.type(torch.FloatTensor)
                    outputs = net(data)
                    labels_list.append(labels)
                    outputs_list.append(outputs)
                    loss_list.append(F.binary_cross_entropy(outputs, labels))
            log.eval_test(epoch, loss_list, labels_list, outputs_list)
    log.close()
    print('Finished Training')
