from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from gsp import gsp_doa, plot_random_signal, get_noiseless_signal
from gnn import GCN, CNN, GraphSignalsDataset, plot_accuracy, gnn_doa, visualize_complex_matrix
from parameters import plots_dir, Amp, f0, N, M
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


def compare_methods(snr_vec, label, gsp_doa_accuracy_vs_snr, cnn_doa_accuracy_vs_snr, gnn_doa_accuracy_vs_snr, interferences_str):
    fig = plt.figure()
    if gsp_doa_accuracy_vs_snr.size > 0:
        plt.plot(snr_vec, gsp_doa_accuracy_vs_snr, label='GSP', linewidth=2)
    if cnn_doa_accuracy_vs_snr.size > 0:
        plt.plot(snr_vec, cnn_doa_accuracy_vs_snr, label='CNN', linewidth=2)
    if gnn_doa_accuracy_vs_snr.size > 0:
        plt.plot(snr_vec, gnn_doa_accuracy_vs_snr, label='GNN', linewidth=2)
    plt.legend()
    plt.xlabel('SNR [dB]')
    if label == '':
        plt.ylabel(f'Accuracy [%]')
    else:
        plt.ylabel(f'Accuracy [%] ({label})')
    plt.xlim(snr_vec[0], snr_vec[-1])
    plt.ylim(45, 101)
    fig.savefig(plots_dir + '/' + interferences_str + '_compare_methods_accuracy.png', dpi=200)
    # plt.show()


def main():
    plt.rcParams['font.size'] = 14
    mkdir_p(plots_dir)

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=5,
                        help='input batch size for testing (default: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--plot-gsp-figs', type=bool, default=True,
                        help='plot GSP figures')
    parser.add_argument('--run-gsp-doa', type=bool, default=True,
                        help='Run Graph Signal Processing DOA estimation')
    parser.add_argument('--run-gnn-doa', type=bool, default=True,
                        help='Run Graph Neural Network DOA estimation')
    parser.add_argument('--run-cnn-doa', type=bool, default=True,
                        help='Run Multi Layer Perceptron DOA estimation')
    args = parser.parse_args()

    use_cuda = False
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    snr_vec_train = np.arange(start=-5, stop=20, step=3)
    snr_vec_test = np.arange(start=-5, stop=20, step=3)
    # snr_vec_test = snr_vec_test[::-1]

    interference1 = get_noiseless_signal(A=0.5*Amp, f=f0, theta=30)
    interference2 = get_noiseless_signal(A=0.5*Amp, f=f0, theta=120)

    train_set = GraphSignalsDataset(num_true_points_per_snr=100,
                                    num_false_points_per_snr=100,
                                    snr_vec=snr_vec_train,
                                    interferences=None)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_set0 = GraphSignalsDataset(num_true_points_per_snr=70,
                                    num_false_points_per_snr=70,
                                    snr_vec=snr_vec_test,
                                    interferences=None)
    test_set1 = GraphSignalsDataset(num_true_points_per_snr=70,
                                    num_false_points_per_snr=70,
                                    snr_vec=snr_vec_test,
                                    interferences=interference1)
    test_set2 = GraphSignalsDataset(num_true_points_per_snr=70,
                                    num_false_points_per_snr=70,
                                    snr_vec=snr_vec_test,
                                    interferences=interference1+interference2)

    test_loader0 = torch.utils.data.DataLoader(
        test_set0,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    # test_loader1 = torch.utils.data.DataLoader(
    #     test_set1,
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test_loader2 = torch.utils.data.DataLoader(
        test_set2,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    print(f'Training set size = {train_set.__len__()}')
    print(f'Test set size = {test_set0.__len__()}')

    gsp_doa_accuracy_vs_snr0 = np.array([])
    cnn_doa_accuracy_vs_snr0 = np.array([])
    gnn_doa_accuracy_vs_snr0 = np.array([])

    gsp_doa_accuracy_vs_snr1 = np.array([])
    cnn_doa_accuracy_vs_snr1 = np.array([])
    gnn_doa_accuracy_vs_snr1 = np.array([])

    gsp_doa_accuracy_vs_snr2 = np.array([])
    cnn_doa_accuracy_vs_snr2 = np.array([])
    gnn_doa_accuracy_vs_snr2 = np.array([])

    ####################################################################################################################
    # Plot data samples
    ####################################################################################################################
    if args.plot_gsp_figs:
        plot_random_signal(test_set0.get_signals(), label=True, snr=14, interference_str='0', theta_str='theta_d', snr_str='high_snr')
        plot_random_signal(test_set0.get_signals(), label=False, snr=14, interference_str='0', theta_str='not_theta_d', snr_str='high_snr')
        plot_random_signal(test_set0.get_signals(), label=True, snr=-5, interference_str='0', theta_str='theta_d', snr_str='low_snr')
        plot_random_signal(test_set0.get_signals(), label=False, snr=-5, interference_str='0', theta_str='not_theta_d', snr_str='low_snr')

        plot_random_signal(test_set1.get_signals(), label=True, snr=14, interference_str='1', theta_str='theta_d', snr_str='high_snr')
        plot_random_signal(test_set1.get_signals(), label=False, snr=14, interference_str='1', theta_str='not_theta_d', snr_str='high_snr')
        plot_random_signal(test_set1.get_signals(), label=True, snr=-5, interference_str='1', theta_str='theta_d', snr_str='low_snr')
        plot_random_signal(test_set1.get_signals(), label=False, snr=-5, interference_str='1', theta_str='not_theta_d', snr_str='low_snr')

        plot_random_signal(test_set2.get_signals(), label=True, snr=14, interference_str='2',  theta_str='theta_d', snr_str='high_snr')
        plot_random_signal(test_set2.get_signals(), label=False, snr=14, interference_str='2',  theta_str='not_theta_d', snr_str='high_snr')
        plot_random_signal(test_set2.get_signals(), label=True, snr=-5, interference_str='2',  theta_str='theta_d', snr_str='low_snr')
        plot_random_signal(test_set2.get_signals(), label=False, snr=-5, interference_str='2',  theta_str='not_theta_d', snr_str='low_snr')
    ####################################################################################################################
    # Multi Layer Perceptron DOA
    ####################################################################################################################
    if args.run_cnn_doa:
        # model = GraphNet().to(device)
        cnn_model = CNN().to(device)
        print(cnn_model)
        optimizer = optim.SGD(cnn_model.parameters(), lr=args.lr, weight_decay=1e-1)
        print('number of trainable parameters: %d' %
              np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in cnn_model.parameters()]))

        train_acc_vec = np.array([])
        test_acc_vec = np.array([])

        for epoch in range(1, args.epochs + 1):
            train_acc = train(args, cnn_model, device, train_loader, optimizer, epoch)
            test_acc = test(args, cnn_model, device, test_loader0)

            train_acc_vec = np.append(train_acc_vec, train_acc)
            test_acc_vec = np.append(test_acc_vec, test_acc)

        plot_accuracy(args, 'CNN', train_acc_vec, test_acc_vec)

        cnn_doa_est_labels_vec0, cnn_doa_true_accuracy_vs_snr0, cnn_doa_false_accuracy_vs_snr0 = gnn_doa(cnn_model, test_set0)
        cnn_doa_accuracy_vs_snr0 = (cnn_doa_true_accuracy_vs_snr0 + cnn_doa_false_accuracy_vs_snr0) / 2
        cnn_doa_est_labels_vec1, cnn_doa_true_accuracy_vs_snr1, cnn_doa_false_accuracy_vs_snr1 = gnn_doa(cnn_model, test_set1)
        cnn_doa_accuracy_vs_snr1 = (cnn_doa_true_accuracy_vs_snr1 + cnn_doa_false_accuracy_vs_snr1) / 2
        cnn_doa_est_labels_vec2, cnn_doa_true_accuracy_vs_snr2, cnn_doa_false_accuracy_vs_snr2 = gnn_doa(cnn_model, test_set2)
        cnn_doa_accuracy_vs_snr2 = (cnn_doa_true_accuracy_vs_snr2 + cnn_doa_false_accuracy_vs_snr2) / 2
        # visualize_complex_matrix(cnn_model.fc1.fc_r.weight.data.numpy() +
        #                          1j * cnn_model.fc1.fc_i.weight.data.numpy(), 'CNN weights')
        # W_cnn = cnn_model.fc1.fc_r.weight.data.numpy() + 1j * cnn_model.fc1.fc_i.weight.data.numpy()
        # print(np.linalg.norm(W_cnn))

        np.set_printoptions(precision=2)
        print("CNN accuracy:")
        print(cnn_doa_accuracy_vs_snr0)
        print(cnn_doa_accuracy_vs_snr1)
        print(cnn_doa_accuracy_vs_snr2)
    ####################################################################################################################
    # Graph Neural Network DOA
    ####################################################################################################################
    if args.run_gnn_doa:
        # model = GraphNet().to(device)
        gnn_model = GCN().to(device)
        print(gnn_model)
        optimizer = optim.SGD(gnn_model.parameters(), lr=args.lr, weight_decay=0)
        print('number of trainable parameters: %d' %
              np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in gnn_model.parameters()]))

        train_acc_vec = np.array([])
        test_acc_vec = np.array([])

        for epoch in range(1, args.epochs + 1):
            train_acc = train(args, gnn_model, device, train_loader, optimizer, epoch)
            test_acc = test(args, gnn_model, device, test_loader0)

            train_acc_vec = np.append(train_acc_vec, train_acc)
            test_acc_vec = np.append(test_acc_vec, test_acc)

        plot_accuracy(args, 'GNN', train_acc_vec, test_acc_vec)

        gnn_doa_est_labels_vec0, gnn_doa_true_accuracy_vs_snr0, gnn_doa_false_accuracy_vs_snr0 = gnn_doa(gnn_model, test_set0)
        gnn_doa_accuracy_vs_snr0 = (gnn_doa_true_accuracy_vs_snr0 + gnn_doa_false_accuracy_vs_snr0) / 2

        gnn_doa_est_labels_vec1, gnn_doa_true_accuracy_vs_snr1, gnn_doa_false_accuracy_vs_snr1 = gnn_doa(gnn_model, test_set1)
        gnn_doa_accuracy_vs_snr1 = (gnn_doa_true_accuracy_vs_snr1 + gnn_doa_false_accuracy_vs_snr1) / 2

        gnn_doa_est_labels_vec2, gnn_doa_true_accuracy_vs_snr2, gnn_doa_false_accuracy_vs_snr2 = gnn_doa(gnn_model, test_set2)
        gnn_doa_accuracy_vs_snr2 = (gnn_doa_true_accuracy_vs_snr2 + gnn_doa_false_accuracy_vs_snr2) / 2

        # visualize_complex_matrix(gnn_model.gcn_layer1.fc_layers[0].fc_r.weight.data.numpy() +
        #                          1j*gnn_model.gcn_layer1.fc_layers[0].fc_i.weight.data.numpy(), 'GCN weights')
        # W_gnn = gnn_model.gcn_layer1.fc_layers[0].fc_r.weight.data.numpy() + 1j * gnn_model.gcn_layer1.fc_layers[
        #     0].fc_i.weight.data.numpy()
        # print(np.linalg.norm(W_gnn))

        np.set_printoptions(precision=2)
        print("GNN accuracy:")
        print(gnn_doa_accuracy_vs_snr0)
        print(gnn_doa_accuracy_vs_snr1)
        print(gnn_doa_accuracy_vs_snr2)
    ####################################################################################################################
    # Graph Signal Processing DOA
    ####################################################################################################################
    if args.run_gsp_doa:
        gsp_doa_est_theta_vec2, gsp_doa_est_labels_vec2, gsp_doa_true_accuracy_vs_snr2, gsp_doa_false_accuracy_vs_snr2 = gsp_doa(test_set2)
        gsp_doa_accuracy_vs_snr2 = (gsp_doa_true_accuracy_vs_snr2 + gsp_doa_false_accuracy_vs_snr2) / 2

        # gsp_doa_est_theta_vec1, gsp_doa_est_labels_vec1, gsp_doa_true_accuracy_vs_snr1, gsp_doa_false_accuracy_vs_snr1 = gsp_doa(test_set1)
        # gsp_doa_accuracy_vs_snr1 = (gsp_doa_true_accuracy_vs_snr1 + gsp_doa_false_accuracy_vs_snr1) / 2

        # gsp_doa_est_theta_vec0, gsp_doa_est_labels_vec0, gsp_doa_true_accuracy_vs_snr0, gsp_doa_false_accuracy_vs_snr0 = gsp_doa(test_set0)
        # gsp_doa_accuracy_vs_snr0 = (gsp_doa_true_accuracy_vs_snr0 + gsp_doa_false_accuracy_vs_snr0) / 2

        np.set_printoptions(precision=2)
        print("GSP accuracy:")
        print(gsp_doa_accuracy_vs_snr0)
        print(gsp_doa_accuracy_vs_snr1)
        print(gsp_doa_accuracy_vs_snr2)

    ####################################################################################################################
    # Compare
    ####################################################################################################################
    compare_methods(test_set0.signals['snr'], '',
                    gsp_doa_accuracy_vs_snr0,
                    cnn_doa_accuracy_vs_snr0,
                    gnn_doa_accuracy_vs_snr0, '0')
    compare_methods(test_set0.signals['snr'], '',
                    gsp_doa_accuracy_vs_snr1,
                    cnn_doa_accuracy_vs_snr1,
                    gnn_doa_accuracy_vs_snr1, '1')
    compare_methods(test_set0.signals['snr'], '',
                    gsp_doa_accuracy_vs_snr2,
                    cnn_doa_accuracy_vs_snr2,
                    gnn_doa_accuracy_vs_snr2, '2')
    print("All done.")


if __name__ == '__main__':
    main()
