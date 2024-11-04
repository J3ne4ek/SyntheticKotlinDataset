# SyntheticKotlinDataset

### Usage

Before using the scripts, you need to install the dependencies, as well as the Kotlin compiler for evaluation

```bash
./setup.sh
```

To create dataset dataset with prob;ems and solutions run 2 scripts. Before using, you need to put the value of the API key into a variable OPENAI_API_KEY
```bash
python dataset_scripts/dataset_creation_gpt.py
python dataset_scripts/filtering_and_splitting.py
```

To benchmark model. You can specify model using parameter --model_path
```bash
python benchmark.py --model_path Salesforce/codegen-350M-mono
```

To fine-tune model. You can specify model_path, dataset_path and output_path
```bash
python train.py
```

### Description
To generate a dataset, I use the open api, as a prompt I pass a request to translate the code into Kotlin, using a certain style of writing code. It is important that the data is the same. Despite this, in some cases the model does not cope, so before dividing I check that the code really starts with a docstring, as indicated in the prompt, if this is true, then I split it into a problem and a solution after defining the first function in the code. In my case, the model resources were limited, so I set the dataset size to 10k, as well as a not very large number of new tokens in the gpt request, to improve the quality of the dataset, these parameters can be corrected. Due to the small window, some tasks also did not fit, if the task only has a condition, then I do not leave it in the dataset, I leave an incompletely generated solution in the dataset, although this can also be changed

Next, using the trainer, I fine-tune the model with the following hyperparameters

| Hyperparameter               | Value             |
|------------------------------|-------------------|
| `warmup_ratio`               | 0.1               |
| `lr_scheduler_type`          | "linear"          |
| `per_device_train_batch_size`| 32                |
| `num_train_epochs`           | 2                 |
| `learning_rate`              | 2.5e-5            |
| `fp16`                       | True              |
| `weight_decay`               | 0.01              |

### Results
Before training, the model does not know Kotlin, since it was taught only in Python, so the metric is below 1%. After training, the model's performance significantly improves. For comparison, I also trained the model on a JetBrains/KExercises dataset with the same parameters, which did not give significant increases in the metric. They are slightly higher, which is due to the large number of tokens in the dataset. Thus, we can conclude that the dataset is of good quality

| Model                                | Pass rate |
|--------------------------------------|---------|
| Baseline                             | 0.621   |
| Fine-tuned with JetBrains/KExercises | 11.180  |
| Fine-tuned with my dataset           | 9.31  |


