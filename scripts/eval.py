import sys
import torch
from logging import warning
from transformers import AutoModelForCausalLM, AutoTokenizer

max_new_tokens = 2048
max_len = 2048
warning(f'defaulting to model_max_length={max_len}')
modelpath="output/eng_savant_pyth"

def generate(model, tokenizer, input_text):
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        input_ids = input_ids.to('cuda')

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                #attention_mask=attention_mask,
                max_length=128,
                temperature=0.6,
                use_cache=True,
                penalty_alpha=0.6,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                #no_repeat_ngram_size=3,
                num_return_sequences=1,
            )
    
        response = tokenizer.decode(generated_ids[:, input_ids.shape[-1] :][0], skip_special_tokens=True)
        return response

def main():
    model = AutoModelForCausalLM.from_pretrained(modelpath, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(
        modelpath, 
        model_max_length=max_len, 
        truncation_side="left", 
        truncation="True"
        )
    
    while True:
         input_text = input("Prompt: ")
         output_text = generate(model, tokenizer, input_text)
         print("Response:",output_text)
         print()
    
if __name__ == '__main__':
    sys.exit(main())