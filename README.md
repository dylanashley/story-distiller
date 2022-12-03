# story-distiller

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project attempts to embed a story into a music playlist by sorting the playlist so that the order of the music follows a narrative arc. The music tracks are fitted to a fixed narrative template based on the output of a machine learning model, which itself distills each track down to its narrative essence. For more information on narrative essence, see [*On Narrative Information and the Distillation of Stories*](https://arxiv.org/abs/2211.12423) by Dylan R. Ashley, Vincent Herrmann, Zachary Friggstad, and Jürgen Schmidhuber.

## Installation

This project is implemented in [Python](https://www.python.org/) and uses models learned with [PyTorch](https://pytorch.org).

To use this project, first install the remaining required packages using pip:
```bash
pip install -r requirements.txt
```

Afterwards, you can directly execute the `__main__.py` file or use the makefile to compile and then install an executable Python zip archive:
```bash
make all
sudo make install
```

To run the program, execute it while passing the audio files as command-line arguments:
```bash
sdi files [files ...] >> playlist.txt
```

If you want to try out a different template, pass the `-t` argument to the program with the template file as an argument. Several learned templates are included in the templates directory:

![templates.jpg](https://github.com/dylanashley/story-distiller/blob/main/templates.jpg?raw=true)
