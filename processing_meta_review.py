'''
Overall process:

1. Processing
2. Tokenizing
3. Uploading
'''

# ==============================================================
# set to True if you want to login and upload to huggingface
want_to_push = False
username = 'sdansdk'
if want_to_push:
    import subprocess

    # Run the shell command :: !huggingface-cli login
    subprocess.run(["huggingface-cli", "login"])
    username = input("Enter your huggingface username: ")

# ==============================================================
import re
import time
import datasets

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

# ==============================================================
# global variables
dataset_name = 'zqz979/meta-review'
model_name = 'EleutherAI/pythia-70m'
max_len = 2048 # this is the limit of their models
batch_size = 1 # due to compute limits on my end

# fetch from huggingface
unprocessed_dataset = datasets.load_dataset(dataset_name)

# tokenizer from the model only 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# a guiding prompt to help the model learn to summarize
text_template = """### Review:
{review}

### Summary:"""

# ==============================================================
def preprocess_text(text):
    '''
    function to preprocess text like removing punctuations and special characters
    input: string
    output: string
    '''
    # Remove punctuation and special characters; only keep "a-zA-Z0-9"
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# ==============================================================
def process_data(split):
    '''
    To process the input and output data
    input: string 
        - split in (train, validation, test)

    output: Dict
    '''
    input_ = []
    output_ = []
    for data_point in unprocessed_dataset[split]:
        # got none in some of the reviews
        try:
            input_.append(text_template.format(review = preprocess_text(data_point['Input'])))
            output_.append(preprocess_text(data_point['Output']))
        except:
            pass
    return {"Input":input_,"Output":output_}

# ==============================================================
# create datasets from processed data dicts
train_dataset = Dataset.from_dict(process_data('train'))
val_dataset = Dataset.from_dict(process_data('validation'))
test_dataset = Dataset.from_dict(process_data('test'))

# Create DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

if want_to_push:
    dataset_dict.push_to_hub(f'{username}/processed_meta_review')
    # to let huggingface reflect the changes
    time.sleep(30)

# ==============================================================
def tokenize_function(examples):
    '''
    To tokenize the input and output data and store their input_ids and attention_mask
    input: Dict
    output: Dict
    '''
    text = examples["Input"][0] + examples["Output"][0]
    # in case it was not done before, setting padding token
    tokenizer.pad_token = tokenizer.eos_token
    # tokenize with padding
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
    )
    # to decide the max length of the sequence
    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        max_len # 2048
    )
    # left truncation
    tokenizer.truncation_side = "left"
    # tokenize with truncation set to max length
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length
    )

    return tokenized_inputs

# load processed dataset from huggingface
finetuning_dataset_loaded = datasets.load_dataset(f"{username}/processed_meta_review")

# Tokenize the dataset using the tokenize_function via map
tokenized_dataset = finetuning_dataset_loaded.map(
    tokenize_function,
    batched=True,
    batch_size=1,
    drop_last_batch=True # to avoid that last uneven batch of leftover examples
)

# ==============================================================
# Add labels to the dataset for easy training via huggingface APIs
tokenized_dataset['train'] = tokenized_dataset['train'].add_column("labels", tokenized_dataset['train']["input_ids"])
tokenized_dataset['validation'] = tokenized_dataset['validation'].add_column("labels", tokenized_dataset['validation']["input_ids"])
tokenized_dataset['test'] = tokenized_dataset['test'].add_column("labels", tokenized_dataset['test']["input_ids"])

if want_to_push:
    tokenized_dataset.push_to_hub('sdansdk/tokenized_meta_review')