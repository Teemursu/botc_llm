# Default configurations

language="english"
learning_rate = 5e-5
#model = "TurkuNLP/gpt3-finnish-large" 
model ="EleutherAI/pythia-1b-v0"
num_train_epochs = 4
per_device_batch_size = 16
output_dir = "output"
output_file = "eng_savant_pyth"
gradient_accumulation_steps = 1
local_rank = None  # Default value for local rank in distributed training
use_lora = False
transformers_cache = ".cache"  # Default cache directory for transformers
dropout = 0.1
prompt_structure = False
