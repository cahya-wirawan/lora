import gc
import os
import sys
import threading
import random
from itertools import chain

import numpy as np
import psutil
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model

# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def main():
    accelerator = Accelerator()
    model_name_or_path = "bigscience/bloomz-560m"
    dataset_name = "twitter_complaints"
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    text_column = "text"
    lr = 5e-5
    num_epochs = 5
    batch_size = 4
    seed = 42
    max_length = 64
    set_seed(seed)
    preprocessing_num_workers=4
    overwrite_cache = False
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    checkpoint_dir = "checkpoint"

    logger = get_logger(__name__)
    dataset = load_dataset("cahya/instructions-all", dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    block_size = tokenizer.model_max_length
    if block_size > 1024:
        block_size = 1024
    def tokenize_function(examples):
        return tokenizer(examples[text_column])

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            # remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=per_device_eval_batch_size
    )

    print(next(iter(train_dataloader)))

    # creating model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )
    accelerator.print(model)

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    for epoch in range(num_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        accelerator.save_state(checkpoint_dir)
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )
        train_epoch_loss = total_loss / len(eval_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        model.eval()
        eval_preds = []
        with TorchTracemalloc() as tracemalloc:
            for _, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
                    )  # synced_gpus=True for DS-stage 3
                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
                preds = accelerator.gather(outputs)
                preds = preds[:, max_length:].detach().cpu().numpy()
                eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )

    accelerator.wait_for_everyone()
    model.push_to_hub(
        "cahya/"
        + f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace("/", "_"),
        state_dict=accelerator.get_state_dict(model),
        use_auth_token=True,
    )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
