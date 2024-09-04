# TP-GMOT: Tracking Generic Multiple Object by Textual Prompt with Motion-Appearance Cost (MAC) SORT

## Overview

This repository contains the implementation for **TP-GMOT: Tracking Generic Multiple Object by Textual Prompt with Motion-Appearance Cost (MAC) SORT**, which was accepted at **ECAI 2024**. TP-GMOT provides an innovative approach to tracking multiple objects in video sequences using textual prompts and a novel motion-appearance cost function.

## Installation

To set up the environment and install the required dependencies, follow these steps:

1. **Create the Conda Environment:**

   ```bash
   conda env create -f environment.yaml
   ```

2. **Activate the Conda Environment:**

   ```bash
   conda activate tpgmot
   ```

3. **Install the Package:**

   ```bash
   pip install -e .
   ```

## Usage

To run the tracking algorithm, use the following command:

```bash
python track.py --source .asset/car.avi --main-object 'red car'
python track.py --source .asset/car.avi --main-object 'red car' --negative-part 'yellow car. blue car'
python track.py --source .asset/car_w_headlights.avi --main-object 'frontal car' --sub-part 'white headlights'
```

- `--source`: Path to the video file.
- `--main-object`: Description of the main object to track.
- `--negative-part`: Descriptions of objects that should not be tracked.

## Configuration

You can adjust the size of the holders for long and short-term memory with the following arguments:

- `--short-mems`: Size of the holder for short-term memory. Default is `3`.
- `--long-mems`: Size of the holder for long-term memory. Default is `9`.

Example usage with custom memory sizes:

```bash
python track.py --source .asset/car.avi --main-object 'red car' --negative-part 'yellow car blue car' --short-mems 5 --long-mems 12
```