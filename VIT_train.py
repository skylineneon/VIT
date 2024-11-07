from model import VIT
from torch.utils.data import DataLoader
from data import MnistDataset
from torch.optim import Adam
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast , GradScaler


class Trainer:


    def __init__(self):

        self._log = SummaryWriter()


        self._net = VIT().cuda() #创建网络模型

        self._train_dataset = MnistDataset("../MLP/datas/train")
        print("train datasets loading finish ...")
        self._train_dataloader = DataLoader(self._train_dataset,batch_size=5000,shuffle=True)

        # self._train_dataloader = MyDataloader("datas/train")
        
        self._test_dataset = MnistDataset("../MLP/datas/test")
        print("test datasets loading finish ...")
        self._test_dataloader = DataLoader(self._test_dataset,batch_size=10000,shuffle=True)

        # self._test_dataloader = MyDataloader("datas/test")

        self._opt = Adam(self._net.parameters(),lr=0.0001,betas=(0.85,0.95),weight_decay=0.001)

        # self._loss_fn = torch.nn.MSELoss()
        # self._loss_fn = MyMSE()

        self._loss_fn = torch.nn.CrossEntropyLoss()
        # self._loss_fn = torch.nn.NLLLoss()


    def __call__(self):

        print("start train ...")

        _scaler = GradScaler()
        
        for _epoch in range(10000000000000):
            
            #训练
            self._net.train()
            _loss_sum = 0.
            for _i,(_data,_target) in enumerate(self._train_dataloader): 
            # for _i in range(len(self._train_dataloader)//5000):

                _data = _data.cuda()
                _target = _target.cuda()

                with autocast(enabled=False,device_type="cuda"):
                    # _data,_target = self._train_dataloader.get_batch(_i,5000)
                    _y = self._net(_data)
                    print(_y.dtype)

                    # _loss = torch.mean((_y - _target)**2)
                    _loss = self._loss_fn(_y,_target)
                
                self._opt.zero_grad()
                _scaler.scale(_loss).backward()
                _scaler.step(self._opt)
                _scaler.update()
                # _loss.backward()
                # self._opt.step()

                _loss_sum += _loss.detach().cpu().item()
            
            print("loss",_loss_sum/len(self._train_dataloader))

            self._log.add_scalar("loss",_loss_sum/len(self._train_dataloader),_epoch)
        
            #验证
            self._net.eval()
            
            _acc_sum = 0
            for _i,(_data,_target) in enumerate(self._test_dataloader):
                _data = _data.cuda()
                _target = _target.cuda()
                _y = self._net(_data)
                _acc_sum += (_y.argmax(-1) == _target).sum()
            
            print(_acc_sum/len(self._test_dataset))
            self._log.add_scalar("acc",_acc_sum/len(self._test_dataset),_epoch)


if __name__ == "__main__":
    train = Trainer()
    train()
