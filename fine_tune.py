'''
overall process:

1. Load dataset
2. Load model
3. Tokenize dataset
4. Fine tune
5. Upload
'''
# !pip install transformers[torch] datasets
# !pip install huggingface_hub

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
import uuid
import torch
import datasets
import transformers

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import Trainer

# decide which device to use
device_count = torch.cuda.device_count()
if device_count > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ==============================================================
# global variables
model_name = 'EleutherAI/pythia-70m'
dataset_path = f'{username}/tokenized_meta_review'
max_len = 2048
max_out_len = 256
batch_size = 1
no_of_epochs = 1

# load dataset
tokenized_dataset = datasets.load_dataset(dataset_path)
train_dataset, validation_dataset, test_dataset = tokenized_dataset['train'], tokenized_dataset['validation'], tokenized_dataset['test']

tokenizer = AutoTokenizer.from_pretrained(model_name) # load the tokenizer for base model;same model tokenizer
tokenizer.pad_token = tokenizer.eos_token # specify what should be the padding token
# load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.to(device)

# ==============================================================
output_dir = "pythia_finetuned_dir"

# this is higher level than the torch and easier to maintain
# check here for more details on each: https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L178
training_args = TrainingArguments(

    # tested with other lrs(1e-3, 1e-4) but this is the best!! that's it
    learning_rate=1.0e-5,

    # Number of training epochs, can't do more than 1; compute restriction and its time taking even on GPU
    num_train_epochs=no_of_epochs,

    # Batch size for training
    per_device_train_batch_size=batch_size,

    # Directory to save model checkpoints
    output_dir=output_dir, # all is abstracted out, even creating output_dir etc

    # Other arguments
    overwrite_output_dir=False, # Overwrite the content of the output directory
    disable_tqdm=False, # Disable progress bars
    eval_steps=480, # Number of update steps between two evaluations
    save_steps=480, # After # steps model is saved
    warmup_steps=1, # Number of warmup steps for learning rate scheduler
    per_device_eval_batch_size=batch_size, # Batch size for evaluation
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    optim="adafactor",
    gradient_accumulation_steps = 4, # Number of updates steps to accumulate the gradients for, before performing a backward/update pass
    gradient_checkpointing=False, # to faster training

    # Parameters for early stopping
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    greater_is_better=False # here lower is better
)

# instantiate the trainer
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset, # validation_dataset
)

# the magic line:: takes time
training_output = trainer.train()

# ==============================================================
save_dir = f'{output_dir}/final'

trainer.save_model(save_dir)
print("Saved model to:", save_dir)

base_model.save_pretrained(f"{output_dir}/final")

# Push the model to the Hugging Face Model Hub
if want_to_push:
    base_model.push_to_hub(f"sdansdk/pythia_70m_finetuned_{uuid.uuid4().time_hi_version}")