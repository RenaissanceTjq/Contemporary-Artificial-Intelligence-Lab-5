from PIL import Image
from torchvision import transforms
import torch.nn.utils.rnn as run_utils
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
label_id = {'positive':0, 'neutral':1, 'negative':2, 'null':3}


def remove_pattern(input_txt, pattern_1, pattern_2):
    r = re.findall(pattern_1, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    # r = re.findall(pattern_2, input_txt)
    # for j in r:
    input_txt = re.sub(pattern_2, ' ', input_txt)

    return input_txt.lower()

class MyDataset(Dataset):
    def __init__(self, path, filename, text_tokenizer, image_transforms, data_type):
        self.path = path
        self.image_transforms = image_transforms
        self.datatype = data_type
        self.error_guid = []
        self.guid_list= []
        self.text_list = []
        self.label_list = []
        self.text_to_id = []
        with open(path+filename, "r", encoding="utf-8") as f:
            data_labels = f.read().split("\n")[1:]
            while True:
                try:
                    data_labels.remove("")
                except:
                    break
            f.close()
        for guid_label in data_labels:
            guid, label = guid_label.split(",")
            count = int(guid)
            if count % 100 == 0:
                print(count)
            label = label_id[label]
            print(label)
            with open(path+'/data/{}.txt'.format(guid), "r", encoding="gb18030") as f:
                txt = f.read()
                f.close()
            self.guid_list.append(guid)
            self.text_list.append(txt)
            self.label_list.append(label)
        for tex in self.text_list:
            tex = np.vectorize(remove_pattern)(tex, '@[\w]*','[^a-zA-Z\'\’]')#预处理工作
            temp = text_tokenizer.tokenize('[CLS]' + tex + '[SEP]')
            self.text_to_id.append(text_tokenizer.convert_tokens_to_ids(temp))
    def __getitem__(self, item):
        path = self.path + '/data/' + str(self.guid_list[item]) + '.jpg'
        image = Image.open(path)
        return self.text_to_id[item], self.image_transforms(image), self.label_list[item]
    def __len__(self):
        return len(self.text_to_id)
def collate(batch_data):
    text_to_id = [torch.LongTensor(data[0]) for data in batch_data]
    image = torch.FloatTensor([np.array(data[1]) for data in batch_data])
    label = torch.LongTensor([data[2] for data in batch_data])
    max_length = max([text.size(0) for text in text_to_id])
    bert_attention_mask = []
    for text in text_to_id:
        text_mask_cell = [1] * text.size(0)
        text_mask_cell.extend([0] * (max_length - text.size(0)))
        bert_attention_mask.append(text_mask_cell[:])
    text_to_id = run_utils.pad_sequence(text_to_id, batch_first=True, padding_value=0)
    return text_to_id, torch.LongTensor(bert_attention_mask), image, label
def get_data(args, path, filename, text_tokenizer, data_type):
    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        )])
    dataset = MyDataset(path, filename, text_tokenizer, transform, data_type)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=True if data_type == 'train' else False, collate_fn=collate, pin_memory=True)
    return data_loader