import sys
import os
import time
import numpy as np
import torch
import dgl
import pickle
import random
import signal
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
# from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import socnavData
sys.path.append('./nets')
from select_gnn import SELECT_GNN




if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def describe_model(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


def evaluate(feats, model, subgraph, labels, loss_fcn, fw, net):
    with torch.no_grad():
        model.eval()
        model.gnn_object.g = subgraph
        model.g = subgraph
        if fw == 'dgl':
            for layer in model.gnn_object.layers:
                layer.g = subgraph
            if net in ['rgcn']:
                output = model(feats.float(),subgraph.edata['rel_type'].squeeze().to(device))
            else:
                output = model(feats.float(),None)
        else:
            if net in [ 'pgat', 'pgcn', 'ptag', 'psage', 'pcheb' ]:
                data = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(device))
            else:
                data = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(device), edge_type=subgraph.edata['rel_type'].squeeze().to(device))
            output = model(data, subgraph)

        a = output.flatten()
        b = labels.float().flatten()
        loss_data = loss_fcn(a.to(device), b.to(device))
        predict = a.data.cpu().numpy()
        got = b.data.cpu().numpy()
        score = mean_squared_error(got, predict)
        return score, loss_data.item()


stop_training = False
ctrl_c_counter = 0


def signal_handler(sig, frame):
    global stop_training
    global ctrl_c_counter
    ctrl_c_counter += 1
    if ctrl_c_counter == 3:
        stop_training = True
    if ctrl_c_counter >= 5:
        sys.exit(-1)
    print('If you press Ctr+c  3 times we will stop saving the training ({} times)'.format(ctrl_c_counter))
    print('If you press Ctr+c >5 times we will stop NOT saving the training ({} times)'.format(ctrl_c_counter))


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# MAIN

def main(training_file, dev_file, test_file, graph_type=None, net=None, epochs=None, patience=None, batch_size=None,
         num_hidden=None, heads=None, gnn_layers=None, nonlinearity=None, residual=None, lr=None, weight_decay=None,
         in_drop=None, alpha=None, attn_drop=None, cuda=None, fw='dgl',index=None, previous_model=None):
    global stop_training

    if nonlinearity == 'relu':
        nonlinearity = F.relu
    elif nonlinearity == 'elu':
        nonlinearity = F.elu

    loss_fcn = torch.nn.MSELoss()  # (reduction='sum')

    print('=========================')
    # print('HEADS', heads)
    # print('OUT_HEADS', num_out_heads)
    print('GNN LAYERS', gnn_layers)
    print('HIDDEN', num_hidden)
    print('RESIDUAL', residual)
    print('inDROP', in_drop)
    print('atDROP', attn_drop)
    print('LR', lr)
    print('DECAY', weight_decay)
    print('ALPHA', alpha)
    print('BATCH', batch_size)
    print('GRAPH_ALT', graph_type)
    print('ARCHITECTURE', net)
    print('=========================')

    # create the dataset
    time_dataset_a = time.time()
    print('Loading training set...')
    train_dataset = socnavData.GenerateDataset(training_file, mode='train', alt=graph_type)
    print('Loading dev set...')
    valid_dataset = socnavData.GenerateDataset(dev_file, mode='valid', alt=graph_type)
    print('Loading test set...')
    test_dataset = socnavData.GenerateDataset(test_file, mode='test', alt=graph_type)
    print('Done loading files')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)
    time_dataset_b = time.time()
    for _ in range(5):
        print(f'TIME {time_dataset_b - time_dataset_a}')

    num_rels = train_dataset.num_rels
    cur_step = 0
    best_loss = -1
    n_classes = num_hidden[-1]
    print('Number of classes:  {}'.format(n_classes))
    num_feats = train_dataset[0][0].features.shape[1]
    print('Number of features: {}'.format(num_feats))
    g = dgl.batch(train_dataset.data)
    num_hidden_layers_rgcn = 3
    num_hidden_layers_gat = 3
    num_hidden_layer_pairs = 3
    K = 10
    #heads = ([num_heads] * gnn_layers) + [num_out_heads]
    # define the model
    model = SELECT_GNN(num_feats,
                 1,            #n_classes
                 num_hidden,
                 gnn_layers,
                 in_drop,
                 nonlinearity,
                 1,
                 net,
                 K,                # sage filters
                 heads,
                 train_dataset.num_rels,
                 train_dataset.num_rels,
                 g,
                 residual,
                 attn_drop,
                 num_hidden_layers_rgcn,
                 num_hidden_layers_gat,
                 num_hidden_layer_pairs
                 )
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # for name, param in model.named_parameters():
    # if param.requires_grad:
    # print(name, param.data.shape)
    if previous_model is not None:
        model.load_state_dict(torch.load(previous_model, map_location = device))

    model = model.to(device)

    for epoch in range(epochs):
        if stop_training:
            print("Stopping training. Please wait.")
            break
        model.train()
        loss_list = []
        for batch, data in enumerate(train_dataloader):
            subgraph, feats, labels = data
            subgraph.set_n_initializer(dgl.init.zero_initializer)
            subgraph.set_e_initializer(dgl.init.zero_initializer)
            feats = feats.to(device)
            labels = labels.to(device)
            if fw == 'dgl':
                model.gnn_object.g = subgraph
                model.g = subgraph
                for layer in model.gnn_object.layers:
                    layer.g = subgraph
                if net in ['rgcn']:
                    logits = model(feats.float(), subgraph.edata['rel_type'].squeeze().to(device))
                else:
                    logits = model(feats.float(), None)
            else:
                model.g = subgraph
                if net in [ 'pgat', 'pgcn', 'ptag', 'psage', 'pcheb' ]:
                    data = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(device))
                else:
                    data = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(device), edge_type=subgraph.edata['rel_type'].squeeze().to(device))

                logits = model(data, subgraph)
            a = logits ## [getMaskForBatch(subgraph)].flatten()
            # print('AA', a.shape)
            # print(a)
            a = a.flatten()
            #print('labels', labels.shape)
            b = labels.float()
            # print('b')
            # print(b)
            b = b.flatten()
            # print('BB', b.shape)
            ad = a.to(device)
            bd = b.to(device)

            loss = loss_fcn(ad, bd)
            optimizer.zero_grad()
            a = list(model.parameters())[0].clone()
            loss.backward()
            optimizer.step()
            b = list(model.parameters())[0].clone()
            not_learning = torch.equal(a.data, b.data)
            if not_learning:
                import sys
                print('Not learning')
                # sys.exit(1)
            else:
                pass
                # print('Diff: ', (a.data-b.data).sum())
            # print(loss.item())
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()
        print('Loss: {}'.format(loss_data))
        if epoch % 5 == 0:
            if epoch % 5 == 0:
                print("Epoch {:05d} | Loss: {:.4f} | Patience: {} | ".format(epoch, loss_data, cur_step), end='')
            score_list = []
            val_loss_list = []
            for batch, valid_data in enumerate(valid_dataloader):
                subgraph, feats, labels = valid_data
                subgraph.set_n_initializer(dgl.init.zero_initializer)
                subgraph.set_e_initializer(dgl.init.zero_initializer)
                feats = feats.to(device)
                labels = labels.to(device)
                score, val_loss = evaluate(feats.float(), model, subgraph, labels.float(), loss_fcn, fw, net)
                score_list.append(score)
                val_loss_list.append(val_loss)
            mean_score = np.array(score_list).mean()
            mean_val_loss = np.array(val_loss_list).mean()
            if epoch % 5 == 0:
                print("Score: {:.4f} MEAN: {:.4f} BEST: {:.4f}".format(mean_score, mean_val_loss, best_loss))
            # early stop
            if best_loss > mean_val_loss or best_loss < 0:
                print('Saving...')
                directory = str(index).zfill(5)
                os.system('mkdir ' + directory)
                best_loss = mean_val_loss
                # Save the model
                torch.save(model.state_dict(), directory + '/SOCNAV.tch')
                params = {'loss': best_loss,
                          'net': net, #str(type(net)),
                          'fw': fw,
                          'gnn_layers': gnn_layers,
                          'num_feats': num_feats,
                          'num_hidden': num_hidden,
                          'graph_type': graph_type,
                          'n_classes': n_classes,
                          'heads': heads,
                          'nonlinearity': nonlinearity,
                          'in_drop': in_drop,
                          'attn_drop': attn_drop,
                          'alpha': alpha,
                          'residual': residual,
                          'num_rels': num_rels,
                          'K': K,
                          'n_hlayers_rgcn': num_hidden_layers_rgcn,
                          'n_hlayers_gat': num_hidden_layers_gat,
                          'n_hlayers_pairs': num_hidden_layer_pairs
                          }
                pickle.dump(params, open(directory + '/SOCNAV.prms', 'wb'))
                cur_step = 0
            else:
                # print(best_loss, mean_val_loss)
                cur_step += 1
                if cur_step >= patience:
                    break
    test_score_list = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, feats, labels = test_data
        subgraph.set_n_initializer(dgl.init.zero_initializer)
        subgraph.set_e_initializer(dgl.init.zero_initializer)
        feats = feats.to(device)
        labels = labels.to(device)
        test_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn, fw, net)[1])
    print("MSE for the test set {}".format(np.array(test_score_list).mean()))
    model.eval()
    return best_loss


if __name__ == '__main__':
    retrain = False
    if len(sys.argv) == 3:
        ext_args = {}
        for i in range(2):
            _, ext = os.path.splitext(sys.argv[i + 1])
            ext_args[ext] = sys.argv[i + 1]
        if '.prms' in ext_args.keys() and '.tch' in ext_args.keys():
            params = pickle.load(open(ext_args['.prms'], 'rb'), fix_imports=True)
            retrain = True

    if not retrain:
        print("If you want to retrain, use \"python3 train.py file.prms file.tch\"")
        best_loss = main('train_set.txt', 'dev_set.txt', 'test_set.txt',
                         graph_type='1',  #
                         net='gatmc',
                         epochs=500,
                         patience=5,
                         batch_size=40,
                         num_hidden=[50, 40, 30, 15, 3], #[25, 20, 20, 15, 10, 3],
                         heads=[15,15, 1, 1, 10], #[15, 15, 10, 8, 8, 6],
                         residual=False,
                         lr=0.0005,
                         weight_decay=0.000000001,
                         nonlinearity='elu',
                         gnn_layers=5, #6,
                         in_drop=0.,
                         alpha=0.12,
                         attn_drop=0.,
                         cuda=True,
                         fw='dgl')
    else:
        params = pickle.load(open(ext_args['.prms'], 'rb'), fix_imports=True)
        best_loss = main('out_train.json', 'out_dev.json', 'out_test.json',
                         graph_type=params['graph_type'],
                         net=params['net'],
                         epochs=500,
                         patience=5,
                         batch_size=15,
                         num_hidden=params['num_hidden'],
                         heads=params['heads'],
                         residual=params['residual'],
                         lr=0.0001,
                         weight_decay=0.000,
                         gnn_layers=params['gnn_layers'],
                         in_drop=params['in_drop'],
                         alpha=params['alpha'],
                         attn_drop=params['attn_drop'],
                         cuda=True,
                         fw=params['fw'],
                         previous_model=ext_args['.tch'])
