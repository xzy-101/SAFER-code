from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
# import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from datasets import load_from_disk
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
import os
os.environ["HF_HUB_HTTP_TIMEOUT"] = "60"
print("HF_DATASETS_CACHE:", os.getenv("HF_DATASETS_CACHE"))
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    # for 8 GPU, the global batch size is 512
    gradient_accumulation_steps: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="Llama-3.2-1B-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=0,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_set_path: Optional[str] = field(
        default="train_dataset",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    eval_set_path: Optional[str] = field(
        default="eval_dataset",
        metadata={"help": "The dir of the subset of the eval data to use"},
    )
    output_path: Optional[str] = field(
        default="output_path",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)
    save_every_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Eval the model every x steps"},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)
# Adjusted according to the base model
# Need to do this for the models that don't have an official pad token.
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print(tokenizer.padding_side)
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length

# Get the dataset
train_path = script_args.train_set_path
eval_path = script_args.eval_set_path
output_name = script_args.output_path

def build_dataset(tokenizer, train_path, eval_path):

    def tokenize(sample):
        sample['positive'] = tokenizer.apply_chat_template(
            sample['chosen'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        sample['negative'] = tokenizer.apply_chat_template(
            sample['rejected'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        tokenized_pos = tokenizer(sample['positive'], truncation=True)
        tokenized_neg = tokenizer(sample['negative'], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        return sample
    ds = load_from_disk(train_path)
    ds = ds["train"] if "train" in ds else ds
    ds = ds.shuffle(seed=42)
    ds = ds.map(tokenize, num_proc=4)
    eval_dataset = load_from_disk(eval_path)
    eval_dataset = eval_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.map(tokenize, num_proc=4)
    train_dataset = ds
    return train_dataset, eval_dataset

train_dataset, eval_dataset = build_dataset(tokenizer, train_path, eval_path)

# Define the trainer
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_strategy="steps",
    save_steps=script_args.save_every_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to='wandb',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16, use_flash_attention_2=True,
)

model.config.use_cache = not script_args.gradient_checkpointing
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names

# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []

        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch

# Define the trainer
def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    result['accuracy'] = np.sum(
        pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
    return result

class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

# Train the model.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length),
)

trainer.train()
trainer.model.save_pretrained(output_name + "/best_checkpoint")
tokenizer.save_pretrained(output_name + "/best_checkpoint")
