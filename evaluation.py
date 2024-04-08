'''
overall process:

1. Load dataset and tokenizer and model (base and finetuned) 
2. Prompt engineering
3. Inference on base and finetuned model for comparison
4. Store comparison
'''

# ==============================================================
import random
import torch
import datasets
import transformers
import pandas as pd

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import Trainer

# ==============================================================
# decide which device to use
device_count = torch.cuda.device_count()
if device_count > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ==============================================================
# global variables
model_name = 'EleutherAI/pythia-70m'
dataset_path = 'sdansdk/tokenized_meta_review'
finetuned_model_path = 'sdansdk/pythia_70m_finetuned'
max_out_len = 256
max_len = 2048 - max_out_len
batch_size = 1

# ==============================================================
def inference(text, model, tokenizer, max_input_tokens=max_len, max_output_tokens=max_out_len):
    '''
    to do inference on the models
    text: string
    model: LLM
    tokenizer: tokenizer
    max_input_tokens: int
    max_output_tokens: int
    '''
    # Tokenize with truncation set to max length
    tokenized_ = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens # 2048
    )

    # Generate with new tokens set to max_output_tokens
    device = model.device
    generated_tokens_with_prompt = model.generate(
    input_ids=tokenized_['input_ids'].to(device),
    max_new_tokens=max_output_tokens,
    )

    # Decode with skiping special tokens
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # Strip the prompt from generated output
    generated_text_answer = generated_text_with_prompt[0][len(text):]

    return generated_text_answer

# =========================================================
# dataset, tokenizer, and model(base and finetuned)
tokenized_dataset = datasets.load_dataset(dataset_path)
test_dataset = tokenized_dataset['test']
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.eval()
base_model.to(device)
finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)
finetuned_model.eval()
finetuned_model.to(device)

# ============================= prompt engineering ===============================
prompt_ = '''Please summarize the key points and insights from the meta reviews of the paper. 
Focus on highlighting the main contributions, strengths, and weaknesses.
Ensure the summary is concise, informative, and captures the essence.
Summarize the meta review in {no_of_words} words or less.
{review}
'''

# =============================== evaluation ===============================

# Initialize list to store comparisons
compare = []

# Select 20 random examples from the range [1, 1000]
random_examples = random.sample(range(1, 1001), 20)

# Iterate over the selected examples
for idx in random_examples:
    que = test_dataset[idx]['Input']
    ans = test_dataset[idx]['Output']
    
    # Generate summaries using fine-tuned and base models
    ans_fine = inference(prompt_.format(review=que, no_of_words=max_out_len), finetuned_model, tokenizer)
    ans_base = inference(prompt_.format(review=que, no_of_words=max_out_len), base_model, tokenizer)
    
    # Store comparison results
    tmp = {'original summary': ans, 'trained model': ans_fine, 'Base Model': ans_base}
    compare.append(tmp)

# Create DataFrame from comparison results
df = pd.DataFrame.from_dict(compare)
df.to_excel('eval_examples.xlsx', index=False)
