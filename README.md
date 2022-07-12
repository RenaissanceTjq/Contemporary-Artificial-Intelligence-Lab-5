#  Contemporary Artificial Intelligence Lab-5

This is the official repository of the lab-5 of the course Contemporary Artificial Intelligence given by Xiang Li in the [School of Data Science and Engineering](http://dase.ecnu.edu.cn/) at [East China Normal University](http://english.ecnu.edu.cn/)

## Setup

This implementation is based on Python3.8 To run the code, you need the following dependencies:

- torchvision==0.11.3
- pytorch==1.10.2
- tokenizers==0.10.3
- numpy==1.21.5
- tqdm==4.59.0
- transformers==4.18.0
- matplotlib==3.3.4

You can simply run

```python
pip install -r requirements.txt
```

## Repository structure

We select some important files for detailed description.

```python
|-- bert/ # pretrained-model:bert-base
    |-- config.json # config setting of pretrained-model:bert-base
    |-- pytorch_model.bin # bert-base model(not uploaded)
    |-- vocab.txt  # tokenizer for bert-base
|-- data/ # experiments for 9 small-scale datasets
    |-- data/ # 4869 jpg-files and 4869 txt-files for train and test
    |-- dev_with_label.txt # dev set after devided
    |-- result.txt # predictions
    |-- test_without_label.txt # test set
    |-- train.txt # training set after devided
    |-- train_with_label.txt # trainign set 
|-- image/ #images geneated by model
    |-- dev_acc.png # accuracy on dev set at diffent epochs
    |-- dev_loss.png # loss on dev set at diffent epochs
|-- my_model/ # model parameters saved during training(not uploaded)
    |-- model.pth # model parameters(not uploaded)
|-- data_loader.py # the function to load data
|-- main.py # the main code of the whole project
|-- model.py # includes all model implementations
|-- train.py # train the model
```

## How to run

1.Make sure [data](https://goo.gl/jgESp4) is ready.  Train the model:

```python
python main.py -do_train -do_eval -dev_size 0.2 
```

2.Generate predictions:

~~~
python main.py -do_test
~~~

3.Test model performance with only image/test input:

~~~
python main.py -image_text_only
~~~



## Attribution

Parts of this code are based on the following repositories:

- [CIMLF](_https://github.com/Link-Li/CLMLF)



