from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from gsp import gsp_doa, plot_random_signal
from gnn import GraphNet, GraphSignalsDataset, plot_accuracy, gnn_doa
from parameters import plots_dir
from utils import mkdir_p


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, ((data_r, data_i), target) in enumerate(train_loader):
        (data_r, data_i), target = (data_r.to(device), data_i.to(device)), target.to(device)
        optimizer.zero_grad()
        output = model(data_r, data_i)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data_r)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    return 100.0*correct/len(train_loader.dataset)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (data_r, data_i), target in test_loader:
            (data_r, data_i), target = (data_r.to(device), data_i.to(device)), target.to(device)
            output = model(data_r, data_i)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')
    return 100.0*correct/len(test_loader.dataset)


def compare_methods(snr_vec, gsp_doa_accuracy_vs_snr, gnn_doa_accuracy_vs_snr):
    fig = plt.figure()
    plt.plot(snr_vec, gsp_doa_accuracy_vs_snr, label='GSP', linewidth=2)
    plt.plot(snr_vec, gnn_doa_accuracy_vs_snr, label='GNN', linewidth=2)
    plt.legend()
    plt.xlabel('SNR [dB]')
    plt.ylabel('Accuracy [%]')
    plt.xlim(snr_vec[0], snr_vec[-1])
    plt.ylim(0, 100)
    fig.savefig(plots_dir + '/compare_methods_accuracy.png', dpi=200)
    plt.show()


def main():
    plt.rcParams['font.size'] = 14
    mkdir_p(plots_dir)

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=5,
                        help='input batch size for testing (default: 5)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--plot-gsp-figs', type=bool, default=False,
                        help='plot GSP figures')
    parser.add_argument('--run-gsp-doa', type=bool, default=False,
                        help='Run Graph Signal Processing DOA estimation')
    parser.add_argument('--run-gnn-doa', type=bool, default=True,
                        help='Run Graph Neural Network DOA estimation')
    args = parser.parse_args()
    use_cuda = False

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    snr_vec = np.arange(start=-10, stop=14, step=2)
    train_set = GraphSignalsDataset(num_true_points_per_snr=1000,
                                    num_false_points_per_snr=1000,
                                    snr_vec=snr_vec)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_set = GraphSignalsDataset(num_true_points_per_snr=250,
                                   num_false_points_per_snr=250,
                                   snr_vec=snr_vec)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    print(f'Training set size = {train_set.__len__()}')
    print(f'Test set size = {test_set.__len__()}')
    ####################################################################################################################
    # prev network
    ####################################################################################################################
    # in_features, out_features = L.shape[1], 2  # number of graph nodes, number of classes (theta_d, not theta_d)
    # # graph_L = torch.tensor(L, dtype=torch.float)
    # max_deg = 2
    # hidden_dim = in_features
    #
    # # Stack two gnn layers as our model
    # # gnn2 = nn.Sequential(
    # #     gnnLayer(L, in_features, hidden_dim, max_deg),
    # #     gnnLayer(L, hidden_dim, out_features, max_deg),
    # #     ComplexToAbs(dim=1),
    # #     nn.LogSoftmax(dim=1)
    # # )
    # gnn2 = GNN(L, in_features, hidden_dim, out_features, max_deg).to(device)
    # print(gnn2)

    ####################################################################################################################
    # Graph Signal Processing DOA
    ####################################################################################################################
    if args.run_gsp_doa:
        gsp_doa_est_theta_vec, gsp_doa_est_labels_vec, gsp_doa_accuracy_vs_snr = gsp_doa(test_set)
        print("GSP accuracy:")
        np.set_printoptions(precision=2)
        print(gsp_doa_accuracy_vs_snr)

    ####################################################################################################################
    # Graph Neural Network DOA
    ####################################################################################################################
    if args.run_gnn_doa:
        model = GraphNet().to(device)
        print(model)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-1)
        print('number of trainable parameters: %d' %
              np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

        train_acc_vec = np.array([])
        test_acc_vec = np.array([])

        if args.plot_gsp_figs:
            plot_random_signal(train_set.get_signals(), label=True, snr=0)
            plot_random_signal(train_set.get_signals(), label=False, snr=0)
            plot_random_signal(train_set.get_signals(), label=True, snr=5)
            plot_random_signal(train_set.get_signals(), label=False, snr=5)

        for epoch in range(1, args.epochs + 1):
            train_acc = train(args, model, device, train_loader, optimizer, epoch)
            test_acc = test(args, model, device, test_loader)

            train_acc_vec = np.append(train_acc_vec, train_acc)
            test_acc_vec = np.append(test_acc_vec, test_acc)

        plot_accuracy(args, train_acc_vec, test_acc_vec)

        gnn_doa_est_theta_vec, gnn_doa_est_labels_vec, gnn_doa_accuracy_vs_snr = gnn_doa(model, test_set)
        print("GNN accuracy:")
        np.set_printoptions(precision=2)
        print(gnn_doa_accuracy_vs_snr)

    if args.run_gsp_doa and args.run_gnn_doa:
        compare_methods(test_set.signals['snr'], gsp_doa_accuracy_vs_snr, gnn_doa_accuracy_vs_snr)

    print("All done.")


if __name__ == '__main__':
    main()
