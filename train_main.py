import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.autograd import Variable 
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from define_model import get_train_loader_hdf5, get_test_loader_hdf5
from define_model import device, train, evaluate, test_on_files, test_from_h5, test_plot_one_file

layout = {
    "ABCDE": {
        "loss": ["Multiline", ["loss/train", "loss/validation"]]    
        },
}



parser = argparse.ArgumentParser(description='Fichier servant a lentrainement du reseau de neuronnes')

parser.add_argument('-model', choices= ['lstm', 'fcn', 'cnn', 'cnn_drop'], required= True,
                    help='Quel modele utiliser')
parser.add_argument('-save_as', type=str, required= True,
                    help='Nom sous lequel sauvgarder le modele')
parser.add_argument('-train', type=int, choices=[0, 1], required= True,
                    help='Pour entrainer ou pour tester')
parser.add_argument('-batch_size', type=int, required= True,
                    help='Taille du batch')
parser.add_argument('-max_epoch', type=int, required= True,
                    help='Epoch max')
parser.add_argument('-dataset_name', type=str, required= True,
                    help='Nom du fichier HDF5 contenant les datasets')

args = parser.parse_args()

model_file = 'models/' + args.save_as + '.pt'
max_epochs = args.max_epoch
batch_size = args.batch_size

writer = SummaryWriter('experiments/' + args.save_as)
writer.add_custom_scalars(layout)




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.input_shape = (25,257)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.padding = nn.ZeroPad2d(self._get_padding())

    def _get_padding(self):
        # Calculate the padding needed to make the output tensor have the same shape as the input tensor
        H, W = self.input_shape
        H_out = ((H - 1) // 2 - 1) * 2 + 3
        W_out = ((W - 1) // 2 - 1) * 2 + 3
        padding_h = (H - H_out) // 2
        padding_w = (W - W_out) // 2
        return (padding_h + 1, padding_w, padding_h, padding_w + 1)

    def forward(self, x):
        x = (x.to(torch.float32)).view(-1, 1, 25, 257)
    
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.upsample1(x)
        x = self.conv4(x)
        x = self.upsample2(x)
        x = self.conv5(x)
        x = self.padding(x)
    
        out = x.view(-1, 125, 257)

        return out


class LSTM(nn.Module):
    def __init__(self):

        super(LSTM, self).__init__()

        self.fc_1 =  nn.Linear(257, 257) #fully connected 1

        self.lstm = nn.LSTM(input_size=257, hidden_size=257,
                          num_layers=2, batch_first=True) #lstm

        self.fc_2 =  nn.Linear(257, 257) #fully connected 2

        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = (x.to(torch.float32)).view(-1, 25, 257)
        out = self.relu(self.fc_1(out))

        h_0 = Variable(torch.zeros(2, out.size(0), 257)).to(device) #hidden state
        c_0 = Variable(torch.zeros(2, out.size(0), 257)).to(device) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(out, (h_0, c_0)) #lstm with input, hidden, and internal state
        out = self.fc_2(output)
        out = self.relu(out)
        out = out.view(-1, 125, 257) #reshaping the data for Dense layer next

        return out

class CNN(nn.Module):
    def __init__(self):

        super(CNN, self).__init__()

        self.c1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 7) #CNN
        self.b1 = nn.BatchNorm2d(num_features = 8)
        self.c2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 5) #CNN
        self.b2 = nn.BatchNorm2d(num_features = 16)
        self.c3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3) #CNN
        self.b3 = nn.BatchNorm2d(num_features = 32)
        self.t1 = nn.ConvTranspose2d(in_channels= 32, out_channels= 16, kernel_size=3)
        self.b4 = nn.BatchNorm2d(num_features = 16)
        self.t2 = nn.ConvTranspose2d(in_channels= 16, out_channels= 8, kernel_size=5)
        self.b5 = nn.BatchNorm2d(num_features = 8)
        self.t3 = nn.ConvTranspose2d(in_channels= 8, out_channels= 1, kernel_size=7)
        self.b6 = nn.BatchNorm2d(num_features = 1)

        self.relu = nn.ReLU()
    
    def forward(self,x):
        
        # Propagate input through LSTM
        x = (x.to(torch.float32)).view(-1, 1, 25, 257)
        out = self.c1(x) #lstm with input, hidden, and internal state
        out = self.relu(self.b1(out))
        out = self.c2(out)
        out = self.relu(self.b2(out))
        out = self.c3(out)
        out = self.relu(self.b3(out))
        out = self.t1(out)
        out = self.relu(self.b4(out))
        out = self.t2(out)
        out = self.relu(self.b5(out))
        out = self.t3(out)
        out = self.relu(self.b6(out))
        out = out.view(-1, 125, 257)
        # out = self.relu(self.fc_1(out))
        # out = self.relu(self.fc_2(out))

        return out

class FCNN(nn.Module):
    def __init__(self):

        super(FCNN, self).__init__()

        self.fc_1 = nn.Linear(257, 512) #fully connected 1
        self.fc_2 = nn.Linear((512), 512) #fully connected 2
        self.fc_3 = nn.Linear(512, 512) #fully connected 2
        self.fc_4 = nn.Linear(512, 512) #fully connected 2
        self.fc_5 = nn.Linear(512, 257) #fully connected last layer
        
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = x.to(torch.float32)

        out = self.fc_1(out)
        out = self.relu(out)

        out = self.fc_2(out)
        out = self.relu(out)

        out = self.fc_3(out)
        out = self.relu(out)

        out = self.fc_4(out)
        out = self.relu(out)

        out = self.fc_5(out)
        # out = self.relu(out)
        return out

    # def __init__(self):
    #     super(FCNN, self).__init__()
    #     self.input_layer = nn.Linear(257, 500)
    #     self.hidden_layer1 = nn.Linear(500, 500)
    #     self.hidden_layer2 = nn.Linear(500, 500)
    #     self.hidden_layer3 = nn.Linear(500, 500)
    #     self.output_layer = nn.Linear(500, 257)
    #     self.activation = nn.ReLU()

    # def forward( self, x):
    #     x = x.view(-1, 257)
    #     output= self.activation(self.input_layer(x))
    #     output = self.activation(self.hidden_layer1(output))
    #     output = self.activation(self.hidden_layer2(output))
    #     #output = self.activation(self.hidden_layer3(output))
    #     output_layer = self.output_layer(output)
    #     final = torch.nn.functional.relu(output_layer)

    #     return final



def simple_test():

    if (args.model == 'lstm'):
        model = LSTM().to(device)
        # summary(model, (2, 125, 257))
    elif(args.model == 'fcn'):
        model = FCNN().to(device)
        # summary(model, (2, 125, 257))
    elif(args.model == 'cnn'):
        model = CNN().to(device)
        # summary(model, (2, 1, 125, 257))
    else:
        model = Autoencoder().to(device)

    test_on_files(model, model_file)
    # test_plot_one_file(model, model_file)

def train_main():
    
    if (args.model == 'lstm'):
        model = LSTM().to(device)
        # summary(model, ((2, 125, 257), (2, 125, 257), (2, 125, 257)))
    elif(args.model == 'fcn'):
        model = FCNN().to(device)
        summary(model, (2, 125, 257))
    elif(args.model == 'cnn'):
        model = CNN().to(device)
        summary(model, (2, 1, 125, 257))
    else:
        model = Autoencoder().to(device)
    


    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(model)
    print(f'\n ========== Nb parametres du modele : {count_parameters(model)} ========== \n')


    train_loader = get_train_loader_hdf5(batch_size, args.dataset_name)
    validation_loader = get_test_loader_hdf5(batch_size, args.dataset_name)
    best_val_loss = None
    best_val_epoch = 0
    max_stagnation = 5
    early_stop = False
    for epoch in range(1, max_epochs + 1):
        start_time = datetime.now()
        # Set model to training mode
        model.train(True)
        epoch_loss = 0.
        # Loop over each batch from the training set
        for data, target in train_loader:

            data = data.to(device)
            target = target.to(device)

            # Zero gradient buffers
            optimizer.zero_grad()

            # Pass data through the network
            output = model(data)
            output = torch.squeeze(output)

            # Calculate loss
            loss = criterion(output, target)
            epoch_loss += loss.item()

            # Backpropagate
            loss.backward()

            # Update weights
            optimizer.step()
        epoch_loss /= len(train_loader.dataset)
        writer.add_scalar("loss/train", epoch_loss, epoch)

        print('Train Epoch: {}, Loss: {:.4f}'.format(epoch, epoch_loss))

        # train(model, train_loader, criterion, optimizer, epoch, log)

        ## VALIDATION
        with torch.no_grad():
            print('\nValidation:')
            # evaluate(model, validation_loader, criterion, epoch, log)
            model.eval()
            loss = 0.
            snr = 0.
            for data, target in validation_loader:
                # scaler = MinMaxScaler(feature_range=(0, 1))
                # data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
                # target = scaler.transform(target.reshape(-1, target.shape[-1])).reshape(target.shape)
                data = data.to(device)
                target = target.to(device)
                
                output = torch.squeeze(model(data))
                # snr += mix.compute_SNR(target, output)
                loss += criterion(output, target).item()
            
            loss /= len(validation_loader.dataset)
            # snr /= len(validation_loader.dataset)
            writer.add_scalar("loss/validation", loss, epoch)
            # writer.add_scalar("loss/val_SNR", snr, epoch)
            print('Average loss: {:.4f}\n'.format(loss))
            if ((best_val_loss is None) or (best_val_loss > loss)):
                best_val_loss, best_val_epoch = loss, epoch
            if (best_val_epoch < (epoch - max_stagnation)):
                # nothing is improving for a while
                early_stop = True

        end_time = datetime.now()
        epoch_time = (end_time - start_time).total_seconds()
        txt = 'Epoch took {:.2f} seconds.'.format(epoch_time)
        print(txt)
        if(early_stop):
            print(f"Stagnation reached")
            break


    torch.save(model.state_dict(), model_file)

def test_main():

    print('Reading', model_file)
    if (args.model == 'lstm'):
        model = LSTM().to(device)
    elif(args.model == 'fcn'):
        model = FCNN().to(device)
    elif(args.model == 'cnn'):
        model = CNN().to(device)
    else:
        model = Autoencoder().to(device)

    model.load_state_dict(torch.load(model_file, map_location = torch.device(device)))
    # model.load_state_dict(torch.load('models/trained_model_fc_500_2_error_0.060_permutati'), map_location = torch.device(device))
    model.to(device)

    test_loader = get_test_loader_hdf5(25, args.dataset_name)

    print('=========')
    print('Simple:')
    with torch.no_grad():
        evaluate(model, test_loader)


if __name__ == '__main__':
    print(args.train)
    if(args.train):
        train_main()
    # test_main()
    simple_test()
    # test_from_h5()