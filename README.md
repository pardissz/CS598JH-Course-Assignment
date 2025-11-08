# CS598JH Assignment: KG-RAG
**Name:** Pardissadat Zahraei
**NetID:** zahraei2

---

## 1. Project Overview

This project evaluates and enhances a Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) framework for biomedical question answering. The goal is to reproduce a baseline, implement three required improvement strategies (Modes 0-3), and then develop a novel, high-performance bonus model (Mode 30) that addresses the baseline's limitations, particularly in context retrieval and reasoning.

---

## 2. Folder Structure

The project is organized as follows:

* `/Data`: Contains all required data files.
* `/Code`: Contains the main Python script `run_mcq_qa.py`.
* `run_and_evaluate.ipynb`: A Google Colab notebook demonstrating how to run the key experiments and evaluate the results.
* `report.pdf`: The final 4-page report.

---

## 3. Setup and Running

There are two ways to run this project:

### A) Google Colab (Recommended)

The file `run_and_evaluate.ipynb` is a Google Colab notebook. It contains all the necessary steps to:
1.  Set up the environment and install packages.
2.  Mount Google Drive to access the `/Data` folder.
3.  Run the key experiments (Modes 0, 1, 2, 3, and 30).
4.  Load the resulting `.csv` files and evaluate the accuracy for the report.

### B) Local Execution

1.  Place all data files in the `/Data` folder.
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt 
    ```
3.  Configure the paths in `/Code/config.yaml` to point to the correct file locations in `/Data` and your desired output directory.
4.  Set your Google API key as an environment variable or as described in the script.
5.  To run an experiment, edit the `MODE` variable at the top of `/Code/run_mcq_qa.py`.
6.  Run the main script from within the `/Code` directory:
    ```bash
    cd Code
    python run_mcq_qa.py "models/gemini-flash-latest"
    ```

---

## 4. Experiment Modes

The `MODE` variable in `run_mcq_qa.py` selects the experiment.

### Required Strategies (for Main Report)
* **MODE = "0"**: Baseline KG-RAG. Retrieves raw text context.
* **MODE = "1"**: Improvement I (JSON). Structures the retrieved context as a JSON object.
* **MODE = "2"**: Improvement II (Knowledge-Enhanced). Appends a prior knowledge suffix (e.g., "Provenance... is useless").
* **MODE = "3"**: Improvement III (Integrated). Combines both the JSON structure and the prior knowledge suffix.

### Bonus Strategies (for Bonus Report)
* **MODE = "30"**: The final bonus model. This is a 6-stage adaptive pipeline that uses multi-strategy retrieval, context quality assessment, adaptive reasoning based on quality, and ensemble voting.
* **MODE = "4" - "28"**: Various other bonus experiments and development iterations (e.g., different prompting, retrieval, or filtering strategies).
* **MODE = "30a" / "30b" / "30c"**: Ablation studies for the Mode 30 pipeline, used to test the independent contributions of multi-retrieval, adaptive reasoning, and ensemble voting.

---
