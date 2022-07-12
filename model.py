import torch.nn.modules as nn
import torchvision.models as cv_models
import torch
from transformers import BertConfig, BertForPreTraining
class TextModel(nn.Module):#Bert
    def __init__(self):
        super(TextModel, self).__init__()
        self.config = BertConfig.from_pretrained('bert/')
        self.model = BertForPreTraining.from_pretrained('bert/', config=self.config)
        self.model = self.model.bert
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, input, attention_mask):
        output = self.model(input, attention_mask=attention_mask)
        return output
class ImageModel(nn.Module):#Resnet50
    def __init__(self):
        super(ImageModel, self).__init__()
        resnet = cv_models.resnet50(pretrained=True)
        resnet.eval()
        self.shape = resnet.fc.in_features
        resnet = list(resnet.children())[:-1]
        self.model =nn.Sequential(*resnet)
class FuseModel(nn.Module):
    def __init__(self):
        super(FuseModel, self).__init__()
        self.text_model = TextModel()
        self.image_model = ImageModel().model
        self.image_shape = ImageModel().shape
        self.image_linear = nn.Sequential(
            nn.Linear(in_features=self.image_shape, out_features=32),
            nn.ReLU(inplace=True)
        )
        self.txt_linear = nn.Sequential(
            nn.Linear(in_features=self.text_model.config.hidden_size, out_features=32),
            nn.ReLU(inplace=True)
        )
        self.fuse = torch.nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=False, dropout=0.6)
        self.linear = nn.Sequential(
                            nn.Dropout(),
                            nn.Linear(in_features=64, out_features=128),
                            nn.ReLU(inplace=True),
                            nn.Dropout(),
                            nn.Linear(in_features=128, out_features=3)
                        )

    def forward(self, input):
        txt, bert_attention_mask, image = input
        txt = self.txt_linear(self.text_model(txt, attention_mask=bert_attention_mask).last_hidden_state[:, 0, :]).unsqueeze(dim=0)
        image = self.image_linear(self.image_model(image).flatten(1)).unsqueeze(dim=0)
        fusion = self.fuse(torch.cat((image, txt), dim=2)).squeeze()
        return self.linear(fusion)