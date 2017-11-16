# FCCapsNet
A PyTorch implementation of FCCapsNet based on IJCAI2018 paper [xxxxx](xxxxx)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c soumith
conda install pytorch torchvision cuda80 -c soumith # install it if you have installed cuda
```
- PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```
- tqdm
```
pip install tqdm
```

## Datasets

### Train、Val Dataset
The train and val datasets are sampled from [VOC2012](http://cvlab.postech.ac.kr/~mooyeol/pascal_voc_2012/).
Train dataset has 16700 images and Val dataset has 425 images.
Download the datasets from [here](https://pan.baidu.com/s/1c17nfeo), and then extract it into `data` directory. Finally run
```
python data_utils.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 3]
```
to generate train and val datasets from VOC2012 with given upscale factors(options: 2、3、4、8).

### Test Dataset
The test dataset are sampled from 
| **Set 5** |  [Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)
| **Set 14** |  [Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests)
| **BSD 100** | [Martin et al. ICCV 2001](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
| **Sun-Hays 80** | [Sun and Hays ICCP 2012](http://cs.brown.edu/~lbsun/SRproj2012/SR_iccp2012.html)
| **Urban 100** | [Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).
Download the dataset from [here](https://pan.baidu.com/s/1nuGyn8l), and then extract it into `data` directory.

## Usage

### Train

```
python -m visdom.server & python train.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 3]
--num_epochs          super resolution epochs number [default value is 200]
```
Visdom now can be accessed by going to `127.0.0.1:8097` in your browser, or your own host address if specified.

If the above does not work, try using an SSH tunnel to your server by adding the following line to your local `~/.ssh/config` :
`LocalForward 127.0.0.1:8097 127.0.0.1:8097`.

Maybe if you are in China, you should download the static resources from [here](https://pan.baidu.com/s/1hr80UbU), and
put them on `~/anaconda3/lib/python3.6/site-packages/visdom/static/`.

### Test
```
python test.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 3]
--model_name          super resolution model name [default value is epoch_3_200.pt]
```
The output high resolution images are on `results` directory.

## Benchmarks
Highest accuracy was 99.57% after 30 epochs. The model may achieve a higher accuracy as shown by the trend of the loss/accuracy graphs below.
<table>
  <tr>
    <td>
     <img src="results/train_loss.png"/>
    </td>
    <td>
     <img src="results/test_loss.png"/>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td>
     <img src="results/train_acc.png"/>
    </td>
    <td>
     <img src="results/test_acc.png"/>
    </td>
  </tr>
</table>

The confusion matrix of the digit numbers are showed below.
<img src="results/confusion_matrix.png"/>

The reconstructions of the digit numbers are showed at right and the ground truth at left.
<table>
  <tr>
    <td>
     <img src="results/ground_truth.jpg"/>
    </td>
    <td>
     <img src="results/reconstruction.jpg"/>
    </td>
  </tr>
</table>

Default PyTorch Adam optimizer hyperparameters were used with no learning rate scheduling. Epochs with batch size of 100 takes ~2 minutes on a NVIDIA GTX 1070 GPU. 

