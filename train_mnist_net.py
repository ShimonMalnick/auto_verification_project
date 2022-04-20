import torch
import torchvision
from torch import nn
import torch.optim as optim
from my_own_net import MyOwnNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def load_data(config):
    t = transforms.Compose(
        [transforms.ToTensor(),
         lambda x: x.flatten()])

    train_set = torchvision.datasets.MNIST('./mnist/train', train=True, download=True, transform=t)
    train_dl = DataLoader(train_set, batch_size=config['bs'], shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST('./mnist/train', train=False, download=True, transform=t)
    test_dl = DataLoader(test_set, batch_size=config['bs'], shuffle=True, num_workers=2)
    return train_dl, test_dl


def get_args():
    parser = ArgumentParser(description='Train a classification network on the mnist digits data set', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bs', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--e', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--save_path', default="./models/mnist_fc_net.pth", help='Where to save the model')
    return vars(parser.parse_args())


def train_net(config, train_dl, criterion):
    model = MyOwnNet()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for e in range(config["e"]):
        running_loss = 0.0
        correct = 0
        for i, data in enumerate(train_dl, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()

            # print statistics
            running_loss += loss.item()
            if ((i + 1) % 20) == 0:  # print every 20 batches
                print(f'[{e + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0
        print(f"Epoch {e + 1} train accuracy: ", correct / len(train_dl.dataset))
    model = model.cpu()
    torch.save(model.state_dict(), config['save_path'])
    return model


def test_net(test_dl, train_dl, model, criterion):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for set_name, dl in [('train', train_dl), ('test', test_dl)]:
            cur_loss = 0
            correct = 0
            for data, target in dl:
                data, target = data.view(data.shape[0], -1).to(device), target.to(device)
                output = model(data)
                cur_loss += criterion(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
            cur_loss /= len(dl.dataset)
            print(f'\n{set_name.title()}' + ' set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                cur_loss, correct, len(dl.dataset),
                100. * correct / len(dl.dataset)))


def load_random_examples(indices_path):
    with open(indices_path, "r") as in_file:
        indices = [int(line.strip()) for line in in_file.readlines()]
    t = transforms.Compose(
        [transforms.ToTensor()])
         # trans.Normalize(0.5, 0.5),
         # lambda x: x.numpy().flatten()])
    test_set = torchvision.datasets.MNIST('./mnist/test', train=False, download=True, transform=t)
    test_set_w_indices = Subset(test_set, indices)
    return test_set_w_indices


def main():
    config = get_args()
    train_dl, test_dl = load_data(config)
    criterion = nn.CrossEntropyLoss()
    model = train_net(config, train_dl, criterion)
    test_net(test_dl, train_dl, model, criterion)


if __name__ == '__main__':
    main()
