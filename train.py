import torch
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

dev_loss_epoch = []
dev_acc_epoch = []
def acc(pred, real):
    pred = torch.max(pred, 1)[1]
    # print(pred)
    # print(real)
    return (pred == real).cpu().numpy().sum() / len(pred)


def get_param(data, input_text=True, input_image=True):
    texts, bert_attention_mask, images, labels = data
    if not input_text:
        texts, bert_attention_mask = torch.zeros_like(texts), torch.zeros_like(bert_attention_mask)
    if not input_image:
        images = torch.zeros_like(images)
    return texts.cuda(), bert_attention_mask.cuda(), images.cuda(), labels.cuda()


def eval_model(model, val_dataloader, loss_fn, device, type):
    with torch.no_grad():
        val_loss_sum, val_acc_sum, step = 0., 0., 0
        for data in tqdm(val_dataloader):
            model.eval()
            if type == 0:
                texts, bert_attention_mask, images, labels = get_param(data)
            elif type == 1:
                texts, bert_attention_mask, images, labels = get_param(data, input_text=False)
            elif type == 2:
                texts, bert_attention_mask, images, labels = get_param(data, input_image=False)
            output = model([texts, bert_attention_mask, images])
            val_loss_sum += loss_fn(output, labels)
            val_acc_sum += acc(output, labels)
            step += 1
        val_loss = val_loss_sum / step
        val_acc = val_acc_sum / step
        temp = val_acc.item()
        dev_acc_epoch.append(temp)
        # print(val_acc)
        temp = val_loss.item()
        dev_loss_epoch.append(temp)
        # print(val_loss)
        return val_loss, val_acc


def train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs, device, save):
    best_acc = 0
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, step = 0., 0., 0
        model.train()
        model.zero_grad()
        for data in tqdm(train_dataloader):
            texts, bert_attention_mask, images, labels = get_param(data)
            output = model([texts, bert_attention_mask, images])
            loss = loss_fn(output, labels)
            step_acc = acc(output, labels)
            loss.backward()
            # print(loss)
            optimizer.step()
            optimizer.zero_grad()
            train_loss_sum += loss
            train_acc_sum += step_acc
            step += 1

        train_loss = train_loss_sum / step
        train_acc = train_acc_sum / step

        val_loss, val_acc = eval_model(model, val_dataloader, loss_fn, device, 0)
        print('Epoch:', '%03d' % epoch,
              'train loss =', '%06f' % train_loss,
              'train acc =', '%06f' % train_acc,
              'val loss =', '%06f' % val_loss,
              'val acc =', '%06f' % val_acc,
              end=' ')

        if save and val_acc > best_acc:
            torch.save(model.state_dict(), './my_model/model.pth')
            best_acc = val_acc
            print('model saved')
        else:
            print()
    paint(dev_loss_epoch, dev_acc_epoch)
    # print("text text_1")
    val_loss, val_acc = eval_model(model, val_dataloader, loss_fn, device, 1)
    # print("text text_2")
    print("If input data only contains text:",
          'val loss =', '%06f' % val_loss,
          'val acc =', '%06f' % val_acc,
          end='\n'
          )#此处没有'\n'会导致这句话在终端无法显示
    # print("text image_1")
    val_loss, val_acc = eval_model(model, val_dataloader, loss_fn, device, 2)
    # print("text image_2")
    print("If input data only contains image:",
          'val loss =', '%06f' % val_loss,
          'val acc =', '%06f' % val_acc,
          end=''
          )
def test_model_on_single_input(model, val_dataloader, loss_fn, optimizer,device):
    model.load_state_dict(torch.load('./my_model/model.pth', map_location=device))
    val_loss, val_acc = eval_model(model, val_dataloader, loss_fn, device, 1)
    print("test-text-only start:")
    print("If input data only contains text:",
          'val loss =', '%06f' % val_loss,
          'val acc =', '%06f' % val_acc,
          end='\n'
          )
    val_loss, val_acc = eval_model(model, val_dataloader, loss_fn, device, 2)
    print("test-image-only start:")
    print("If input data only contains image:",
          'val loss =', '%06f' % val_loss,
          'val acc =', '%06f' % val_acc,
          end=''
          )
def paint(dev_loss_epoch,dev_acc_epoch):
    # print("output:")
    # print(dev_loss_epoch)
    # print(dev_acc_epoch)
    y_1 = []
    y_2 = []
    for i in dev_loss_epoch:
      y_1.append(i)
    for i in dev_acc_epoch:
      y_2.append(i)
    x = [i for i in range(len(dev_loss_epoch))]
    fig = plt.figure()
    plt.title("accuracy on dev set")
    plt.plot(x, y_2, '-o', label='accuracy')
    plt.xlabel("epoch number")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig('./image/dev_acc.png')
    fig = plt.figure()
    plt.title("loss on dev set")
    plt.plot(x, y_1, '-o', label='loss')
    plt.xlabel("epoch number")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('./image/dev_loss.png')
def predict(model, test_loader, device):
    y_pred = []
    label_dic = {0: "positive", 1: "neutral", 2: "negative"}
    model.load_state_dict(torch.load('./my_model/model.pth', map_location=device))
    with torch.no_grad():
        model.eval()
        for data in tqdm(test_loader):
            texts, bert_attention_mask, images, _ = get_param(data)
            output = model([texts, bert_attention_mask, images])
            y_pred.extend(torch.max(output, 1)[1].cpu())
            with open('./data/test_without_label.txt', 'r', encoding="utf-8") as f:
                data = f.read().split("\n")[1:]
                f.close()
            with open('./data/result.txt', 'w', encoding="utf-8") as f:
                f.write("guid,tag\n")
                pred, data_pos = 0, 0
                while pred < len(y_pred) and data_pos < len(data):
                    guid = data[data_pos].split(",")[0]
                    f.write(guid + ',' + label_dic[int(y_pred[pred])] + '\n')
                    data_pos += 1
                    pred += 1
                f.close()
