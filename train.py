from dataset import *
import torch
from torch.utils.data import DataLoader
from model import *
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "/home/MyServer/data/fruit/train_data"
test_dir = "/home/MyServer/data/fruit/test_data"
train_list = os.listdir(data_dir)
label_txt = "/home/MyServer/My_Code/MachineLearning/fruit_classification/label.txt"
epoch = 5
lr = 1e-4
batchsize = 128
model_path = "/home/MyServer/My_Code/MachineLearning/fruit_classification"

def train_fn(model, data_dir, fruit_names, epoch, lr, batchsize, model_path):
    train_dataset = FruitDataset(data_dir, fruit_names)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr)

    for i in range(epoch):
        train_loader = DataLoader(train_dataset, batchsize, shuffle=True)
        model.train()
        train_accuracy = 0
        total__train_loss = 0
        loop = tqdm(train_loader)
        for data in loop:
            imgs, labels = data
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            y_pre = model(imgs)
            loss = loss_fn(y_pre, labels)
            optim.zero_grad()   # 设定优化的方向
            loss.backward()     # 从最后一层损失反向计算到所有层的损失     
            optim.step()        # 更新权重

            train_accuracy += (y_pre.argmax(1) == labels).sum()
            total__train_loss += loss.item()
            acc = ((y_pre.argmax(1) == labels).sum() / len(labels)).item()

            loop.set_description(f"Epoch[{i}/{epoch}]")
            loop.set_postfix(loss=loss.item(), acc=acc) # 等号左边的名字随便取
        print("######acc:{}%".format((float(train_accuracy)/float(41322))*100))
        torch.save(model.state_dict(), model_path+"/fruit.pth")

def test_fn(model, data_dir, fruit_names, batchsize):
    test_dataset = FruitDataset(data_dir, fruit_names)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(DEVICE)

    test_loader = DataLoader(test_dataset, batchsize, shuffle=True)
    # model.eval()
    test_accuracy = 0
    total__test_loss = 0
    loop = tqdm(test_loader)
    model.eval()
    for data in loop:
        imgs, labels = data
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        y_pre = model(imgs)
        loss = loss_fn(y_pre, labels)
        loss.backward()     # 从最后一层损失反向计算到所有层的损失     

        test_accuracy += (y_pre.argmax(1) == labels).sum()
        total__test_loss += loss.item()
        acc = ((y_pre.argmax(1) == labels).sum() / len(labels)).item()

        loop.set_postfix(loss=loss.item(), acc=acc) # 等号左边的名字随便取

    print("######acc:{}%".format((float(test_accuracy)/float(13877))*100))

if __name__ == "__main__":
    model = ResNet101(img_channels=3, num_classes=81).to(DEVICE)
    fruit_names = read_labels(label_txt)
    train_fn(model, data_dir, fruit_names, epoch, lr, batchsize, model_path)
    model.load_state_dict(torch.load(model_path+"/fruit.pth"))
    test_fn(model, test_dir, fruit_names, batchsize)

    # transforms = A.Compose([
    #                 A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], 
    #                             max_pixel_value=255,),   # 标准化，归一化
    #                 ToTensorV2()        # 转为tensor
    #             ])
    # img = np.array(Image.open("/home/MyServer/data/fruit/test_data/Peach/5_100.jpg").convert("RGB"))
    # augmentations = transforms(image=img)
    # img = augmentations["image"]
    # img = img.unsqueeze(0)
    # img = img.to(DEVICE)
    # model.eval()    # 一定要加
    # y_pre = model(img)
    # fruit_pre = fruit_names[y_pre.argmax(1).item()]
    # print(fruit_pre)