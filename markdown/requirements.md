# Requirements

The entire project was built in a Windows machine, but it could be easily reproduced in a linux machine. What is really crucial is having access to a dedicated GPU. In the root of this repo a [requirements.txt](https://github.com/Salah-Akil/emotion-recognition/blob/main/requirements.txt) file is included for installing the python dependencies.

I used `Anaconda` as my python manager, and had to build `open-cv` from source in order to make it use the GPU, since the standard `open-cv` installed with `pip` or `conda` does not use the dedicated graphic card.