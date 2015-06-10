# DRL

**[DRL](http://drlearner.org/)** is an implementation of the deep
reinforcement learning algorithm, described in [Playing Atari with Deep
Reinforcement Learning](http://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) paper
by [DeepMind](http://deepmind.com/).

The **DRL** project is based on the [TNNF](http://tnnf.readthedocs.org/en/latest/) - Tiny Neural Net Framework, which uses [Theano](https://github.com/Theano/Theano) library for GPU computing.
Despite the fact that **DRL** shows better performance on GPU rather than on CPU,
it can be launched on CPU (but for learning purposes such approach is not recommended
due to low learning speed).

**DRL** was tested on [Ubuntu 14.04](http://www.ubuntu.com/) and
[CentOS 7](http://www.centos.org/) on **GPU** and **CPU** (for CPU - only playback).

**Games**: [Breakout](https://en.wikipedia.org/wiki/Breakout_(video_game))
([video](http://youtu.be/T58HkwX-OuI)) and
[Space Invaders](https://en.wikipedia.org/wiki/Space_Invaders)

## Installation
To install **DRL** on Ubuntu, please, perform the next steps:
```bash
sudo apt-get install python-pil python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git libsdl1.2-dev libsdl-image1.2-dev libsdl-gfx1.2-dev python-matplotlib libyaml-dev
sudo pip install -U numpy
sudo pip install -U pillow==2.7.0
sudo pip install Theano
```

Now install **pylearn2**:
```bash
cd ~/ # Or any other directory where you want the pylearn2 project to be stored
git clone https://github.com/lisa-lab/pylearn2.git
cd ./pylearn2
sudo python setup.py develop
sudo chown -R $USER ~/.theano # to run the demo without sudo
```

To install **CUDA**, please, follow the instructions on the
[CUDA installation webpage](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html).

Then navigate to the **DRL** directory and run:
```bash
cd DRL
bash ./install.sh
```

To launch the learning process, run the following command from *src* directory:
```bash
bash gpuRun.sh main_exp.py
```

### Trying other ROMs

First 

To change the game for DRL Agent training, navigate to **src\DRL-master\src\ale** directory and open  ale.py on editing 
Change destination "*.bin" for def __init__(ale_game_ROM) (default  is '../emulators/ale_0_4/roms/breakout.bin',  Example: '../emulators/ale_0_4/roms/all/space_invaders.bin')
 
Next

Navigate to **src\DRL-master\src\** directory and open  play.py on editing. 
Change the Model to be used  by changing  self.n_net.loadModel  in  def __init__(self).   (Example: Change ('./models/AS.model.weights.NN.NMem.1') to ('./models/AS.model.weights.NN.NMem.2'), You may also change the path here if needed)

Attention!
For Space Invaders Game change ale(frames_to_skip) to 5 (instead of 6) in play.py , main_exp.py and human_play.py files.

For the detailed instructions and descriptions, please, navigate to the **DRL**
[website](http://drlearner.org/) and download the project documentation.
