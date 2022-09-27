import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, in_c=1):
        super(CNNModel, self).__init__()

        # model = Sequential()
        # TODO: yes/no?
        # do we put all of below into Sequential
        # self.model = nn.Sequential()

        self.conv1 = nn.Conv2d(in_c, 16, kernel_size=3)
        #self.conv2 = nn.Conv2d(16, 32, kernel_size=3)

        """self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
            nn.Flatten(), ## yes/no?
            nn.Linear(512 * 5 * 5, 1024), # ????
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.7), ## too much dropout maybe ???
            nn.Linear(1024, 110), # ?????? 
        )"""



        # add conv2D
        # model.add(Conv2D(16, (3, 3), input_shape=(224, 224, 3), 
        # kernel_initializer='random_uniform', activation='relu'))
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3))

        # model.add(Conv2D(32, (3, 3), activation='relu'))
        # TODO: is number of channels still=1?
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)

        # model.add(MaxPooling2D(pool_size=(2, 2)))
        self.maxpool1 = nn.MaxPool2d(2)

        # model.add(Conv2D(64, (3, 3), activation='relu'))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)

        # model.add(MaxPooling2D(pool_size=(2, 2)))
        self.maxpool2 = nn.MaxPool2d(2)

        # model.add(Conv2D(128, (3, 3), activation='relu'))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)

        # model.add(BatchNormalization())
        # TODO: is this 1D or 2D batch normalization?
        self.batch_norm1 = nn.BatchNorm2d(128)

        # model.add(MaxPooling2D(pool_size=(2, 2)))
        self.maxpool3 = nn.MaxPool2d(2)

        # model.add(Dropout(0.3))
        self.dropout1 = nn.Dropout(0.3)

        # model.add(Conv2D(256, (5, 5), activation='relu'))
        self.conv5 = nn.Conv2d(128, 256, kernel_size=5)

        # model.add(MaxPooling2D(pool_size=(2, 2)))
        self.maxpool4 = nn.MaxPool2d(2)
        
        # model.add(Conv2D(512, (5, 5), activation='relu'))
        self.conv6 = nn.Conv2d(256, 512, kernel_size=5)

        # model.add(BatchNormalization())
        self.batch_norm2 = nn.BatchNorm2d(512)

        # model.add(MaxPooling2D(pool_size=(2, 2)))
        self.maxpool5 = nn.MaxPool2d(2)

        #model.add(Dropout(0.5))
        self.dropout2 = nn.Dropout(0.5)

        # model.add(Flatten())
        # TODO: do i flatten in forward?

        # model.add(Dense(1024, activation='relu'))
        # TODO: https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412/2
        self.dense1 = nn.Linear(512 * 3 * 3, 1024) # TODO: 10?

        # model.add(BatchNormalization())
        self.batch_norm3 = nn.BatchNorm1d(1024)

        # model.add(Dropout(0.7))
        self.dropout3 = nn.Dropout(0.7)

        # model.add(Dense(110))
        # TODO: https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412/2
        self.dense2 = nn.Linear(1024, 110) # TODO: 10?   

        ##self.relu = nn.ReLU() 

        """ some example
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        """

    def forward(self, x):
        """ some example
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        """
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, kernel_size=2)
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = F.max_pool2d(x, kernel_size=2)
        #print(x.shape)
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = F.max_pool2d(self.batch_norm1(x), kernel_size=2)
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv5(x)), kernel_size=2)
        x = F.relu(self.conv6(x))
        x = self.batch_norm2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout2(x)

        #print(x.shape)
        # flatten
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = F.relu(self.dense1(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        return x

    
    def backward(self, X, y):
        pass


    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size()[0], -1)
        
        """
        https://stackoverflow.com/questions/53953460/how-to-flatten-input-in-nn-sequential-in-pytorch/56771143#56771143
        """