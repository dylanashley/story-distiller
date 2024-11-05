# story-distiller

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project attempts to embed a story into a music playlist by sorting the playlist (i.e., sequencing it) so that the order of the music follows a narrative arc. The music tracks are fitted to a fixed narrative template based on the output of a machine learning model, which itself distills each track down to its narrative essence. For more information on narrative essence and how it generalizes to other forms of media, see *On the Distillation of Stories for Transferring Narrative Arcs in Collections of Independent Media* by Dylan R. Ashley, Vincent Herrmann, Zachary Friggstad, and Jürgen Schmidhuber.


## Installation

This project is implemented in [Python](https://www.python.org/) and uses models learned with [PyTorch](https://pytorch.org).

Before installation, first, ensure you have a recent version of Python and pip, then clone the repository using the `--sparse` option:
```bash
git clone --sparse git@github.com:dylanashley/story-distiller.git
```

Afterwards, install the Python dependencies using pip:
```bash
pip install -r requirements.txt
```

To use the full suite of file types supported by the tool, it is also necessary to separately install the [ffmpeg](https://ffmpeg.org/) tool by following the instructions for your operating system provided on [their website](https://ffmpeg.org/download.html).

You can now run the tool by directly executing the `__main__.py` file, or—if you're using Linux or macOS—you can use the makefile to compile and then install an executable Python zip archive:
```bash
make all
sudo make install
```


## Command-line Tool Usage

To run the program, execute it while passing the audio files as command-line arguments:
```bash
sdistil files [files ...] >> playlist.txt
```

[librosa](https://librosa.org/doc/latest/index.html) is used to process the audio files, so most common audio file types are supported.

If you want to try out a different template, pass the `-t` argument to the program with the template file as an argument. Several learned templates are included in the templates directory:

![templates.jpg](https://github.com/dylanashley/story-distiller/blob/main/templates.jpg?raw=true)


## Web App Usage

To run the web app, simply execute the `app.py` file:
```bash
./app.py
```


## Extras

In addition to the above, this repository also includes all the code needed to reproduce the results presented in *On the Distillation of Stories for Transferring Narrative Arcs in Collections of Independent Media* by Dylan R. Ashley, Vincent Herrmann, Zachary Friggstad, and Jürgen Schmidhuber. In particular, it includes
- the scripts needed to learn the PyTorch models for extracting the narrative essence from either music albums (`album_extractor.py`) or movie frames (`movie_extractor.py`) and compute the lower bounds for the mutual information of the different features,
- the scripts for learning a set of template curves from scalar descriptions of items in a set of collections (`template_learner.py`),
- the code that can fit the scalar descriptions of items in a collection to a given template curve (`fit_values` in `__main__.py`),
- the preprocessed album data used to train the original PyTorch models for the music albums (`data/`), and
- the learned PyTorch models and template curves (`results/`).

Note that to obtain the preprocessed data and the learned PyTorch models and template curves, you will have to clone the repository without using the `--sparse` option:
```bash
git clone git@github.com:dylanashley/story-distiller.git
```

At this time, the raw data used to train the movie frame extractor is not possible to release due to copyright issues. However, `movie_extractor.py` should work on any similar collection with minimal changes. Preprocessing to extract CLIP features from images can be done straightforwardly using the [official implementation](https://github.com/openai/CLIP).
