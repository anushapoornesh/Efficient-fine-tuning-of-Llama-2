import copy
import json
import logging
import time
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from dataclasses import dataclass

from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama


IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        torch.tensor(tokenizer.encode(text, bos=True, eos=True)) for text in strings
    ]
    input_ids = labels = [tokenized for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.ne(-1).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources, targets, tokenizer):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len-1] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path, tokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path, "r") as f:
            list_data_dict = json.load(f)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=-1
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(-1),
        )


def make_supervised_data_module(tokenizer, data_path):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def print_params(model: nn.Module):
    trainable = 0 # number of trainable parameters
    untrainable = 0 # number of untrainable parameters
    trainable_size = 0 # size in mb of trainable parameters

    # print("Layers:")
    for name, param in model.named_parameters():
        # print(f"- {name} of size {param.size()} -> {'trainable' if param.requires_grad else 'untrainable'}")

        if param.requires_grad: # if the parameter requires a gradient
            trainable += param.numel() # increment the number of trainable parameters
            trainable_size += param.numel() * param.element_size() # increment the size of trainable parameters
        else:
            untrainable += param.numel() # increment the number of untrainable parameters

    print(f"\nTrainable Parameters: {trainable}")
    print(f"Untrainable Parameters: {untrainable}")
    print(f"Total Parameters: {trainable + untrainable}")
    print(f"Percent Trainable: {100 * trainable / (trainable + untrainable)}%")
    psize = trainable_size / 1024**2
    print(f"Size of trainable parameters: {psize:.2f} MB")


def train():

    torch.manual_seed(1)

    model_path = "/project/saifhash_1190/llama2-7b/consolidated.00.pth"
    tokenizer_path = "/project/saifhash_1190/llama2-7b/tokenizer.model"
    data_path = "alpaca_data_dummy.json"

    # load model
    checkpoint = torch.load(model_path, map_location="cpu")
    model_args = ModelArgs()
    # model_args.n_layers = 2  # for debugging purposes we only use 1 layer
    # torch.set_default_tensor_type(torch.cuda.HalfTensor) # for training we use fp32 weights
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=False)
    model.to("cuda")
    
    # move the model to the device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # load tokenizer
    tokenizer = Tokenizer(tokenizer_path)

    # create dataloader
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=data_path)
    dataloader = torch.utils.data.DataLoader(
        data_module["train_dataset"],
        batch_size=1,
        collate_fn=data_module["data_collator"],
        shuffle=True,
    )

    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    print_params(model) # print the number of parameters

    # prepare optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    iters_to_accumulate = 8

    mem = torch.cuda.memory_allocated(device)
    print("Memory usage before training:", mem/1024**2, "MB\n")

    start_time = time.time()
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to("cuda")
            labels = batch['labels'].to("cuda")

            for i in range(iters_to_accumulate):
                # with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled=True):
                with torch.cuda.amp.autocast():
                    logits = model(input_ids)

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_logits = shift_logits.view(-1, 32000)
                    shift_labels = shift_labels.view(-1)
                    loss = criterion(shift_logits, shift_labels)
                
                scaler.scale(loss).backward()
                total_loss += loss.item()

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        average_loss = total_loss / (iters_to_accumulate * len(dataloader))

        print("Epoch:", epoch, "Loss:", average_loss)
        c = 0
        for i in torch.cuda.mem_get_info():
            if c==0:
                ua = i
                print("Unallocated Memory:",i/1024**2, "MB")
                c=c+1
            else:
                ia = i
                print("Total Allocated Memory:",i/1024**2, "MB")
        print("Maximum Memory Usage:", torch.cuda.max_memory_allocated()/1024**2, "MB\n")
    end_time = time.time()
    print("Training time:", (end_time-start_time)/60,"minutes")

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "How can we reduce air pollution?",
        "Generate a poem that expresses joy",
        "Describe a time when you had to make a difficult decision",
        "Generative AI models are ",
        "Ashoka was the greatest ruler ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        """A letter of resignation:

         Dear team,
          
           I just """
    ]

    model.eval()
    results = model.generate(tokenizer, prompts, max_gen_len=64, temperature=0.6, top_p=0.9)

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n========================================================================\n")

    torch.save(model.state_dict(), 'fine-model.pth')

if __name__ == "__main__":
    train()