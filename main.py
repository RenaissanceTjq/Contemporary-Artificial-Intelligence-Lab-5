import argparse
import os
import numpy as np
import data_loader
import model
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from train import train_model,test_model_on_single_input,predict
import torch.nn.modules as nn
from torch.optim import AdamW
import warnings
def split_dev_set(folder_name,file_name,dev_size = 0.1):
    print(folder_name)
    print("\n")
    with open(os.path.join(folder_name,file_name),'r',encoding="utf-8") as f:
        train_data = f.read().split("\n")[1:]
        print(len(train_data))
        labels = np.zeros(len(train_data))
        print(len(labels))
    train_data,dev_data,train_label,dev_label = train_test_split(train_data,labels,test_size=dev_size)
        # print(len(train_data))
        # print(len(train_label))
        # print(len(dev_data))
        # print(len(dev_label))
        # f.close()
    # print(folder_name)
    # print(os.path.join(folder_name, 'train_with_label'))
    with open (folder_name+"/train_with_label.txt" 'w', encoding='utf-8') as f:#将训练集进行划分，之后训练和验证时分别读取
        f.write("guid,tag\n")
        for i in train_data:
            f.write(str(i)+"\n")
        f.close()
    with open (folder_name+"/dev_with_label.txt", 'w', encoding='utf-8') as f:
        f.write("guid,tag\n")
        for i in dev_data:
            f.write(str(i)+"\n")
        f.close()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-do_train', action='store_true', default=True, help="Whether to run training.")
    parser.add_argument('-do_eval', action='store_true', default=True, help="Whether to run eval on dev set.")
    parser.add_argument('-do_test', action='store_true', default=False, help="Whether to run eval on the dev set.")
    parser.add_argument('-dev_size', type=float, default='0.1', help="Size of dev set (0-1)")
    # parser.add_argument('-text_data', action='store_true', default=True, help="input data contains text data?")
    # parser.add_argument('-image_data', action='store_true', default=True, help="input data contains image data?")
    parser.add_argument('-image_text_only', action='store_true', default=False, help='test model with only text/image input?')
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert/vocab.txt')
    loss_function = nn.CrossEntropyLoss().cuda()
    my_model = model.FuseModel().cuda()
    device = torch.device('cuda')
    optimizer = AdamW(my_model.parameters(), lr=0.00002)
    # split_dev_set("./data", "train.txt", dev_size=args.dev_size)#训练前划分数据集，已划分后注释即可
    if args.do_train is True:
        if args.do_eval is True:
            train_loader = data_loader.get_data(args, "./data/", 'train_with_label.txt', tokenizer, data_type="train")
            dev_loader = data_loader.get_data(args, "./data/", 'dev_with_label.txt', tokenizer, data_type="dev")

        else:
            train_loader = data_loader.get_data(args, "./data/", 'train.txt', tokenizer, data_type="train")
            dev_loader = None

        if args.image_text_only is True:
            test_model_on_single_input(my_model, dev_loader, loss_function, optimizer, device)
        else:
            train_model(my_model, train_loader, dev_loader, loss_function, optimizer, 15, device, './')
    else:
        if args.do_test is True:
            test_loader = data_loader.get_data(args,"./data", 'test_without_label.txt', tokenizer, data_type='test')
            predict(my_model, test_loader, device)
        else:
            print("Please at least train or test your model")