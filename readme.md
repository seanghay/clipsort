## `clipsort` - Organize image files with zero-shot classification

Categorize photos into it corresponding provided label with [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16).

### Basic Usage

Create a new Anaconda environment

```shell
conda create -n clipsort python==3.8
conda activate clipsort
pip install -r requirements.txt
```


Run the inference

```shell
python clipsort.py data/*.jpg --output ./output/ --labels food menu logo
```

Result

```
output
├── food
│   ├── image-11.jpg
│   ├── image-3.jpg
│   ├── image-4.jpg
│   ├── image-5.jpg
│   ├── image-6.jpg
│   ├── image-7.jpg
│   ├── image-8.jpg
│   └── image-9.jpg
├── logo
│   ├── image-10.jpg
│   └── image-2.jpg
└── menu
    └── image-1.jpg
```


## Options

```
usage: clipsort [-h] [-o OUTPUT] -l LABELS [LABELS ...] [-w] [-t TAKE] [-m MODEL] files [files ...]

Sort unstructured photos with CLIP

positional arguments:
  files                 Path to files

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output folder
  -l LABELS [LABELS ...], --labels LABELS [LABELS ...]
  -w, --overwrite       Overwrite exsting file in the output dir
  -t TAKE, --take TAKE  Number of final labels to save
  -m MODEL, --model MODEL
```
