# Human Abstraction and Reasoning Corpus (H-ARC)

This repository contains the H-ARC dataset and preliminary analyses reported in our paper [H-ARC: A Robust Estimate of Human Performance on the Abstraction and Reasoning Corpus](https://arxiv.org/abs/2409.01374).

Participant responses, natural language descriptions, errors and state space graphs can all be explored visually on our [project webpage](https://arc-visualizations.github.io/index.html).

H-ARC consists of action by action traces of humans solving ARC tasks from the both the training and evaluation sets using an interface and setup similar to François Chollet's initial proposal. The original dataset can be found [here](https://github.com/fchollet/ARC-AGI).

## Citing our work

```
@article{legris2024harcrobustestimatehuman,
      title={H-ARC: A Robust Estimate of Human Performance on the Abstraction and Reasoning Corpus Benchmark},
      author={Solim LeGris and Wai Keen Vong and Brenden M. Lake and Todd M. Gureckis},
      year={2024},
      journal={arXiv preprint arxiv:2409.01374}
      url={https://arxiv.org/abs/2409.01374},
}
```

## Getting started

### Setting up the Python Environment

1. Ensure you have Python 3.10 or later installed on your system.

2. Clone this repository to your local machine:

   ```bash
   gh repo clone le-gris/h-arc
   cd h-arc
   ```

3. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate the virtual environment:

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source .venv/bin/activate
     ```

5. Install the required packages using pip and the requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

### Extracting the dataset

The H-ARC dataset is provided as a zip archive in the `data` folder. To extract it:

1. Navigate to the project root directory if you're not already there.

2. Use the following command to extract the dataset:
   - On Windows:
     ```bash
     tar -xf data/h-arc.zip -C data
     ```
   - On macOS and Linux:
     ```bash
     unzip data/h-arc.zip -d data
     ```

After extraction, you should see several CSV files in the `data` folder.

## Dataset

The H-ARC dataset consists of several CSV files containing different aspects of human performance on ARC tasks.

All files are in CSV format. The main files include:

- `clean_data.csv` / `clean_data_incomplete.csv`: All collected data from complete / incomplete participant data
- `clean_errors.csv` / `clean_errors_incomplete.csv`: All unique errors on each task and their counts from complete/incomplete participant data
- `clean_summary_data.csv` / `clean_summary_data_incomplete.csv`: Attempt by attempt summary data for complete/incomplete participant data
- `clean_feedback_data.csv`: Participant feedback
- `clean_demographics_data.csv`: Demographic information
- `clean_withdraw_data.csv`: Withdrawal information

For more detailed information about the dataset, see [Dataset description](data/dataset_description.md).

## Analyses

### Notebooks

## Processing Kaggle Submission

Follow these steps to process a Kaggle submission file:

1. Create the necessary directories:

   ```bash
   mkdir -p data/kaggle_solutions/claude3_5-langchain
   ```

2. Visit the following webpage:
   [Claude 3.5 Langchain ARC Submission](https://www.kaggle.com/code/gregkamradt/using-frontier-models-on-arc-agi-via-langchain/output)

3. Download the `submission.json` file from the webpage into the `data/kaggle_solutions/claude3_5-langchain` directory.

4. Run the `kaggle_submision_to_csv.py` script with the appropriate submission ID:
   ```bash
   python src/kaggle_submision_to_csv.py --submission_id claude3_5-langchain
   ```

This will process the JSON file and create a CSV file in the same directory.

## License

This dataset is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
