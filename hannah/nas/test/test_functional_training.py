import torch
import torchvision
import torchvision.transforms as transforms
from hannah.nas.functional_operators.executor import BasicExecutor
from hannah.nas.functional_operators.op import Tensor

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch.nn.functional as F

from hannah.models.capsule_net_v2.models import search_space

if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    training_set = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=False)

    num_classes = 10

    input = Tensor(name='input',
                   shape=(32, 1, 28, 28),
                   axis=('N', 'C', 'H', 'W'))

    # net = test_net(input)
    net = search_space("net", input)
    net.sample()
    model = BasicExecutor(net)
    model.initialize()

    loss_fn = F.cross_entropy

    optimizer = torch.optim.SGD(list(model.parameters()), lr=0.01, momentum=0.9)
    # print(executor.parameters()[0][0][0])

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model.forward(inputs)
            # print(net.operands[1].data[0][0])

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        # model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        # model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model.forward(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        #     model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        #     torch.save(model.state_dict(), model_path)

        epoch_number += 1

