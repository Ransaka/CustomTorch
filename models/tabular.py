import torch
import torch.nn as nn
import sklearn.metrics as metrics

class TabularModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers, num_hidden, dropout_rate, use_batch_norm,activation):
        super(TabularModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, num_hidden))
            else:
                layers.append(nn.Linear(num_hidden, num_hidden))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(num_hidden))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(num_hidden, output_size))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x, weights):
            output = self.model(x)
            if weights is not None:
                return output * weights
            else:
                return self.model(x)
    
    def fit(self, dataloader, optimizer, criterion, device, epochs):
        # Loop over the specified number of epochs
        best_precision = 0
        best_state_dict = self.state_dict()
        
        for epoch in range(epochs):
            # Loop over the data in the dataloader
            for data, labels, weights in dataloader:
                data, labels, weights = data.to(device), labels.long().to(device), weights.to(device)
                optimizer.zero_grad()
                outputs = self.forward(data, weights)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # After each epoch, calculate the precision, recall, and accuracy
            labels = dataloader.dataset.dataframe['label']
            predictions = self.predict(dataloader, device)
            predictions = torch.tensor(predictions  > 0.5,dtype=torch.int16)
            precision = metrics.precision_score(labels, predictions)
            recall = metrics.recall_score(labels, predictions)
            accuracy = metrics.accuracy_score(labels, predictions)

            if precision > best_precision:
                best_precision = precision
                best_state_dict = self.state_dict()

            # Log the precision, recall, and accuracy for the epoch
            print(f'Epoch {epoch+1}: Precision = {precision:.4f}, Recall = {recall:.4f}, Accuracy = {accuracy:.4f}')

        # After all epochs are complete, load the best performing model
        self.load_state_dict(best_state_dict)
        
    def predict(self, dataloader, device):
        self.eval()
        with torch.no_grad():
            outputs = []
            for data,_,_ in dataloader:
                data = data.to(device)
                output = self.forward(data, weights=None)
                outputs.append(output.cpu())
            return torch.cat(outputs)