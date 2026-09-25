# EEG Analysis of the Berger Effect

## Overview

This project presents an independent analysis of electroencephalography (EEG) data using Python and MNE-Python. The analysis investigates the classic Berger effect: the change in posterior alpha-band activity between eyes-closed and eyes-open conditions.

This personal project was undertaken as an introduction to EEG signal processing. It provided practical experience with data loading, preprocessing, spectral analysis, and visualisation, while developing a foundation for future work in neural signal processing and closed-loop neuromodulation.

## Objectives

The main objectives of this project were to:

- Develop a reproducible EEG analysis workflow.
- Gain practical experience using MNE-Python.
- Prepare and preprocess a publicly available EEG dataset.
- Investigate changes in neural activity between eyes-open and eyes-closed conditions.
- Identify and visualise the Berger effect.
- Develop a foundation for further work in neural signal processing and closed-loop neuromodulation.

## Analysis workflow

The notebook includes the following stages:

1. Loading the EEG dataset.
2. Inspecting the raw EEG recording and metadata.
3. Preprocessing the data.
4. Selecting relevant channels and time periods.
5. Separating eyes-open and eyes-closed conditions.
6. Analysing frequency-domain activity.
7. Comparing alpha-band activity between conditions.
8. Visualising the results.
9. Interpreting the findings in the context of the Berger effect.

## Main finding

The analysis found that alpha-band activity, particularly over posterior electrodes, is stronger in this subject during the eyes-closed condition than during the eyes-open condition.

This pattern is consistent with the Berger effect, in which posterior alpha activity is typically more prominent when the eyes are closed and attenuated when the eyes are open.

## Repository contents

- `EEG analysis pipeline.ipynb` — Main Jupyter notebook containing the analysis.
- `EEG analysis pipeline.html` — HTML export of the notebook for easier reading.
- `requirements.txt` — Python packages required to run the notebook.
- `data/subject_01.mat` —   Example EEG recording used in the analysis.

## Technologies

The project was developed using:

- Python
- MNE-Python
- NumPy
- SciPy
- Matplotlib
- Pandas
- Jupyter Notebook

## Use of AI

Generative AI was used for proofreading and providing general programming guidance. The analysis, code, interpretation of results, and final decisions were completed and reviewed by the author.

## Installation

Clone the repository:

```bash
git clone [https://github.com/aryanmorarji/EEG-analsis-pipeline.git](https://github.com/aryanmorarji/EEG-analysis-pipeline.git)
cd EEG-analysis-pipeline
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Alternatively, open the `.ipynb` file directly in Visual Studio Code with the Python and Jupyter extensions installed.

Run the cells in order from beginning to end.

## Dataset

This project uses the EEG Alpha Waves Dataset, which contains EEG recordings from 20 subjects during eyes-open and eyes-closed resting-state conditions. 

The dataset is available at:

[EEG Alpha Waves Dataset](https://doi.org/10.5281/zenodo.2605110)

The full dataset is not included in this repository. However, the recording used for the initial analysis (subject_01) is included as an example input. Additional participants can be downloaded from the official dataset source and processed using the same workflow.

## Dataset instructions

If you would like to try passing another participant's data through the workflow:

1. Download the EEG Alpha Waves Dataset from [Zenodo](https://doi.org/10.5281/zenodo.2605110).
2. Extract the downloaded files into the location expected by the notebook.
3. Confirm that the filenames and folder structure match the paths used in the notebook.
4. Run the notebook cells in order from beginning to end.

## Reproducibility

The notebook documents the main steps used to process and analyse the EEG Alpha Waves Dataset. To reproduce the analysis:

1. Clone this repository.
2. Install the dependencies listed in `requirements.txt`.
3. For the included example recording, confirm that the file is in the expected `data/` directory.
4. To analyse another participant, download the relevant file  and place them in the same directory structure.
5. Open `EEG analysis pipeline.ipynb`.
6. Run the cells in order from beginning to end.

The dataset source, access instructions, and citation are provided in the Dataset section.

## Limitations

This project is intended as a personal and educational introduction to EEG analysis. The results should therefore be interpreted with the following limitations in mind:

- The analysis uses a small subsection of a publicly available dataset rather than data collected  for this project.
- The analysis represents only the signal-processing component of a complete closed-loop neuromodulation system.

## Future development

Potential extensions of this project include:

- Applying more advanced artefact-removal methods.
- Comparing additional EEG frequency bands.
- Investigating alternative spatial patterns.
- Investigating temporal changes in alpha-band activity.
- Analysing individual differences between participants utilisning more available metadata (Age, fatigue)
- Exploring real-time EEG processing.
- Investigating how extracted neural features could inform closed-loop stimulation.

## Author

Aryan Morarji  
MEng Biomedical Engineering  
Imperial College London

LinkedIn: [linkedin.com/in/aryan-morarji-b452a0303](https://linkedin.com/in/aryan-morarji-b452a0303)

## References

### Berger effect

Berger, H. (1929). Über das Elektrenkephalogramm des Menschen. *Archiv für Psychiatrie und Nervenkrankheiten*, 87, 527–570. https://doi.org/10.1007/BF01797193

### Dataset and associated resources

Cattan, G., Rodrigues, P. L. C., Congedo, M., & Jutten, C. (2017). *EEG Alpha Waves Dataset*. GIPSA-lab. Zenodo. [https://doi.org/10.5281/zenodo.2605110](https://doi.org/10.5281/zenodo.2605110)

The associated example code and analysis resources are available in the [py.ALPHA.EEG.2017-GIPSA repository](https://github.com/plcrodrigues/py.ALPHA.EEG.2017-GIPSA).

### Software

Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience, 7*, 267. https://doi.org/10.3389/fnins.2013.00267

[MNE-Pthon documentation](https://mne.tools/stable/index.html)

