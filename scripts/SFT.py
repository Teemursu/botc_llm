import sys
import os
import torch
import numpy as np
from logging import warning
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from data_preprocessing import BOTC_Dataloader
import utils.config as conf
#print(os.environ.get('CUDA_PATH'))
torch.cuda.empty_cache()
model_max_length = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA:",torch.cuda.is_available())

class PromptMaskingDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        data = super().__call__(features, return_tensors)

        end_of_prompt_id = self.tokenizer.sep_token_id
        for i in range(len(data['labels'])):
            eop_indices = np.where(data['labels'][i] == end_of_prompt_id)[0]
            if len(eop_indices) > 0:
                # TODO this should really be eop_indices[0]+1 but that
                # would mask the eop which would mess up the current
                # logic for separating the prompt from the output
                data['labels'][i,:eop_indices[0]] = -100
            else:
                warning('missing eop in labels')
        return data

def filter_by_length(datasetdict, max_length):
    for k in datasetdict:
        dataset = datasetdict[k]
        filtered = dataset.filter(lambda e: len(e['input_ids']) <= max_length)
        orig_length = len(dataset['input_ids'])
        filt_length = len(filtered['input_ids'])
        if filt_length < orig_length:
            warning(
                f'filtered {k} from {orig_length} to {filt_length} '
                f'({filt_length/orig_length:.1%}) by max_length {max_length}'
            )
            datasetdict[k] = filtered

    return datasetdict

def preprocess(data, tokenizer, prompt_structure): 
    prompts = data['prompt']
    contexts = data['context']
    responses = data['response']
    end_of_prompt = tokenizer.sep_token
    end_of_text = tokenizer.eos_token
    
    combined = []
    for prompt, context, response in zip(prompts, contexts, responses):
            if not context or context.isspace():
                input_i = prompt
            combined.append(input_i + end_of_prompt + response + end_of_text)

    tokenized = tokenizer(combined)
    return tokenized

def main():
    training_args = TrainingArguments(
        output_dir=conf.output_dir,
        evaluation_strategy="steps",
        eval_steps=20,
        learning_rate=conf.learning_rate,
        per_device_train_batch_size=conf.per_device_batch_size,
        gradient_accumulation_steps=conf.gradient_accumulation_steps,
        per_device_eval_batch_size=16,
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=5,
        num_train_epochs=conf.num_train_epochs,
        weight_decay=0.01,
        optim='adamw_hf',
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
    )

    #print(training_args)

    config = AutoConfig.from_pretrained(conf.model,cache_dir=".cache")
    c_dict = config.to_dict()
    c_dict["hidden_dropout"]=conf.dropout
    c_dict["attention_dropout"]=conf.dropout
    c_dict["use_cache"]=False
    config.update(c_dict)
    model = AutoModelForCausalLM.from_pretrained(conf.model,config=config,cache_dir=".cache").to(device)
    tokenizer = AutoTokenizer.from_pretrained(conf.model, config=config, do_lower_case = False)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<|endofprompt|>'})
    model.resize_token_embeddings(len(tokenizer))
    
    dataloader = BOTC_Dataloader(conf.language)
    dataset = dataloader.get_dataset_dict()
    dataset = dataset.map(
         lambda d: preprocess(d, tokenizer, conf.prompt_structure),
            batched=True
        )

    print("Filtering by length")
    dataset = filter_by_length(dataset, model_max_length)

    data_collator = PromptMaskingDataCollator(
            tokenizer=tokenizer,
            mlm=False
        )
    print("Size of training data", len(dataset['train']))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    
    if conf.use_lora:
        trainer.model.save_pretrained(os.path.join(conf.output_dir, conf.output_file))
    else:
        trainer.save_model(os.path.join(conf.output_dir, conf.output_file))

    eval_results = trainer.evaluate(dataset['evaluation'])

    print('Model:', conf.model)
    print('Learning rate:', conf.learning_rate)
    print('batch size:', conf.per_device_batch_size)
    print('Gradient accumulation steps:', conf.gradient_accumulation_steps)
    print('Evaluation results:', eval_results['eval_loss'])

if __name__ == '__main__':
    sys.exit(main())