# taken from https://github.com/clcarwin/sphereface_pytorch

import torch
import torch.nn as nn


class sphere20a(nn.Module):
    def __init__(self,classnum=10574,feature=False):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)

        return x


if __name__ == '__main__':
    sphereface = sphere20a()
    sd = torch.load('C:\\Users\\Administrator\\Desktop\\University\Work\\AdversarialMask\\face_recognition\\sphereface_torch\\weights\\sphereface_20.pth')
    # del sd['fc6.weight']
    # with open('C:\\Users\\Administrator\\Desktop\\University\Work\\AdversarialMask\\face_recognition\\sphereface_torch\\weights\\sphereface_20.pth', 'wb') as f:
    #     torch.save(sd, f)
    sphereface.load_state_dict(sd)
    print('x')