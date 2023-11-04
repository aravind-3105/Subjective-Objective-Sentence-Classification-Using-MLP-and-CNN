# CNN Classifier

This folder contains code for various CNN-based tasks in Section 5

## Files

1. **cnn.py**: This script is responsible for running the primary CNN-based tasks for sections 5.1 and 5.2. It has two main functions:
   - `train_model()`: This function is used to train models and can be used to observe overfitting during training.
   - `parameter_search()`: It iterates through different parameter options and saves the results in a text file to identify the best-performing configuration.

2. **model.py**: This file contains the model definitions for both frozen and unfrozen embeddings. It is integral to the functioning of `cnn.py`.

3. **closest_word_meaning.py**: This module extracts word meanings from the kernels of the CNN models. Note that this script should only be executed after the model has been trained and saved.

4. **A2_Starter.py**: This script was provided in the assignment and contains preprocessing functions for your convenience.

## Running `cnn.py`

To run `cnn.py`, use the following command:

```bash
python3 cnn.py
```

You can specify the following arguments:

- `--type_of_run` (default: "train"): Use this to define the type of run, which can be "train."
- `--n1` (default: 40): Set the value for n1.
- `--k1` (default: 5): Set the value for k1.
- `--n2` (default: 40): Set the value for n2.
- `--k2` (default: 3): Set the value for k2.
- `--batch_size` (default: 16): Define the batch size.
- `--lr` (default: 0.0001): Set the learning rate.
- `--epochs` (default: 30): Specify the number of epochs.
- `--intervals` (default: 5): Determine the intervals for saving results during parameter search.
- `--freeze` (default: "yes"): Specify whether to freeze embeddings ("yes" or "no").
- `--save` (default: "no"): Specify whether to save the model ("yes" or "no").

## Note

The actual model files are not included in this folder due to space constraints. You should train and save the model using `cnn.py`, and then you can use `closest_word_meaning.py` to extract word meanings from the trained model.

Please make sure you have the necessary dependencies and datasets in place before running the code. If you encounter any issues or need further assistance, feel free to reach out.