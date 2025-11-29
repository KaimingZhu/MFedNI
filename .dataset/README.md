# 📚 Dataset Generating in MFedNI

ℹ️ **Introduction:** This file describes how we generate modal incomplete datasets.

### 📚 What datasets have been provided?

We provide two different datasets here.

- **HAR**: `./HAR/origin/`
- **DEAP**: `./DEAP/origin/`

Both of them have been normalized and modal-complete.

### 🏃 How to generate datasets for Federated Learning (FL) evaluation?

In summary, we generate them by introducing four continuous steps.

1. **Incompletion**: transpose it to a modal-incomplete dataset.
2. **Window-Sliding**: resample each datapoint via window sliding.
3. **Division**: Divide subsets for each client in FL (IID/Non-IID).
4. **Splitting**: perform a train-test split for each client's dataset.


We have provided the script to perform them for each dataset. Scripts share naming by their purpose.

Here is an example to perform HAR dataset generation. The way to perform the DEAP dataset generation is similar to it.

```bash
# Please first locate the current path in bash.
# e.g. (a validation check with command `pwd` might output):
#    ~/MFedNI/.dataset
cd HAR

# 1. Incompletion
python make_modal_incomplete_dataset.py

# 2. Window-sliding
python reshape_with_sliding_window.py

# 3. Division
python divide_subsets.py

# 4. Splitting
python train_test_split.py
```

### 🌟 What's more?

The customized hyper-params are now hard-coded in each script, including:

- $\alpha$ of Dirichlet distributions.
- Switching between the Non-IID distribution and the IID distribution.
- The amount of data is modal-incomplete.
- The ratio to split the train-set and test-set.
- The window size for resampling.
- ...

Please refer to each script; they are defined at the beginning of each script. 

You could customize them to meet specific requirements.

