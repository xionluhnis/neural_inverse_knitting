# Inverse Neural Knitting

![Teaser](images/teaser.jpg)

This repository contains code for the paper **"Neural Inverse Knitting: From Images to Manufacturing Instructions"** (http://proceedings.mlr.press/v97/kaspar19a.html).
You can browse the dataset, results and more on the [project page](http://deepknitting.csail.mit.edu).

## Installing dependencies
Several scripts assume a Linux environment with Bash, though it may work on MacOS and/or Windows with CigWIN (but was not tested there).

In general, you need python 3, with several dependencies that can easily be installed with `pip`:

```
pip install -r requirements.txt
```

We suggest doing so in a fresh virtual environment:

```
virtualenv -p python3 env
. env/bin/activate
```

## Downloading dependencies

### Dependencies for inference only
You will need at least a model.
See the pre-trained [model page](http://deepknitting.csail.mit.edu/models/).

For the base network, you can use [MILCE](http://deepknitting.csail.mit.edu/models/experiment-real-milce.tar.gz).
For the full `refiner + img2prog` setup, we suggest [alpha=0.5](http://deepknitting.csail.mit.edu/models/RFI-a0.5.tar.gz) or the [complex,alpha=0.5](http://deepknitting.csail.mit.edu/models/RFI-complex-a0.5.tar.gz), which generated the best results in the paper.

For the refiner network, you will need to download vgg16 in the vgg model directory.

```
# download vgg16.npy and vgg19.npy
cd model/tensorflow_vgg/
./download.sh
```

### Dependencies for training

You will need to download the dataset:

```
./dataset.sh
```

which gets extracted into the folder `dataset`.


## Inference

Inference can be done with

```
./infer.sh -g 0 -c path_to/RFI-complex-a0.5 img1.jpg [img2.jpg ...]
```

where

* `-g 0` is to select GPU 0
* `-c checkpoint/RFI-complex-a0.5` is to use the selected model
* `img1.jpg` is an input image (can use a list of these)

This produces png outputs with same file names.

**Input**
![input](images/Cable2_046_16_0_back.jpg)

**Output**
![output](images/Cable2_046_16_0_back-prog.png)

### Scale detection

At inference, you can specify the `-v` argument to output the average maximum softmax value of the output, which we use in the supplementary to automatically detect the best scale.
For example:

```
./infer.sh -g 0 -c checkpoint/RFI_complex_a0.5 -v img1.jpg
```

A sample output ends with
```
...
Generator variables: 1377554
 [*] Loading checkpoints...
 checkpoint/RFI_complex_a0.5/_lr-0.0005_batch-2/FeedForwardNetworks-150000
  [*] Load SUCCESS
  1 input-img (conf: m=0.767297, s=0.186642)

  Processing Done!
```

where for each image of the list `m` is the mean confidence, and `s` the standard deviation.

### Rendering programs

The repository ships with a pre-trained renderer simulating what the proprietary renderer does.
This is a simple image translation network trained using the mapping from instructions to renderings.

You can render a pattern instruction with

```
CUDA_VISIBLE_DEVICES="0" python3 ./render.py myprog.png [prog2.png ...]
```

where 

* `--output_dir=path_to_dir` can be used to specify the output directory
* `CUDA_VISIBLE_DEVICES=0` is to select the first GPU only
* `myprog.png` is a program output (from `infer.sh`), or a list of these

**Input**
![input](images/Cable2_046_16_0_back-prog.png)

**Output**
![output](images/Cable2_046_16_0_back-rend.png)

### Visualizing programs

We provide a visualization script to make program outputs more easily interpretable.

```
python3 ./test/visualize.py myprog.png [prog2.png ...]
```

will generate files `${file}_viz.png` using the same symbols and colors as shown in the paper.

**Input**
![input](images/Cable2_046_16_0_back-prog.png)

**Output**
![output](images/Cable2_046_16_0_back-viz.png)

## Training from scratch

You should make sure you have downloaded the dataset. You also probably want to download the vgg npy files (see dependencies).

The training script goes through `run.sh` which passes further parameters to `main.py`.
For example, to train the complex RFI network:

```
./run.sh -g 0 -c checkpoint/RFINet_complexnet --learning_rate 0.0005 --params discr_img=1,bvggloss=1,gen_passes=1,bloss_unsup=0,decay_steps=50000,decay_rate=0.3,bunet_test=3 --weights loss_D*=1.0,loss_G*=0.2
```

For the base `img2prog` network, use

```
./run.sh -g 0 -c checkpoint/img2prog --params use_resnet=1,use_rend=0,use_tran=0
```

The code has many different types of network architectures that we tried (and some may or may not make sense anymore).
See the code to figure out what parameters can be tuned, notably see `model/m_feedforw.py` -- the file where the network decision are made for training and testing.

**Note**: the `-c` parameter is a directory path for the named checkpoint. You can / should use your own for training.
The only time it really matters is for inference, when the checkpoint must exist.

## Testing

The test scripts are in `test`.
They require the dataset.

Given a checkpoint, you can create the evaluation data for that checkpoint with `test/eval_checkpoint.sh`.
The test inference results will be generated in a subdirectory `eval` of the checkpoint directory.
Then, these will be used to create renderings and be copied together in the result folders with the checkpoint name.

To create the ground truth results, use
```
./test/results/create_gt.sh
```

## References

If you use this code or system, please cite our paper:

```
@InProceedings{pmlr-v97-kaspar19a,
  title =   {Neural Inverse Knitting: From Images to Manufacturing Instructions},
  author =  {Kaspar, Alexandre and Oh, Tae-Hyun and Makatura, Liane and Kellnhofer, Petr and Matusik, Wojciech},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  pages =   {3272--3281},
  year =    {2019},
  editor =  {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume =  {97},
  series =  {Proceedings of Machine Learning Research},
  address = {Long Beach, California, USA},
  month =   {09--15 Jun},
  publisher = {PMLR},
  pdf =     {http://proceedings.mlr.press/v97/kaspar19a/kaspar19a.pdf},
  url =     {http://proceedings.mlr.press/v97/kaspar19a.html},
}
```
