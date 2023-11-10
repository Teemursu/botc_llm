import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import re  # Import the 're' module for regular expressions
from transformers import DataCollatorForLanguageModeling
import torch

class BOTC_Dataloader:
    def __init__(self, language, csv_file_path="data/botc_data.csv", model_name="TurkuNLP/bert-base-finnish-cased-v1", train_size=1.0, batch_size=4):
        self.df = pd.read_csv(csv_file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_size = train_size
        self.batch_size = batch_size  # Default batch size is set to 4
        self.train_dataset = None
        self.test_dataset = None
        self.valid_dataset = None
        self.language = language
        

    def create_dataset(self, data):
        # Generate the prompts using the provided 'create_prompt' method
        prompts = [self.create_prompt(row['english'], row[self.language]) for _, row in data.iterrows()]
        
        # Set the original sentences as the response
        responses = data[self.language].tolist()

        # Create a new dataset using the prompts and responses
        dataset_dict = {
            "prompt": prompts,
            "context": [""] * len(prompts),
            "response": responses
        }
        dataset = Dataset.from_dict(dataset_dict)
        
        return dataset

    def extract_info(self, english_sent, finnish_sent) -> dict:
        # Extract all words with the first letter uppercase (potential role names)
        english_uppercase_words = re.findall(r'\b[A-Z][a-z]+\b', english_sent)
        finnish_uppercase_words = re.findall(r'\b[A-Z][a-z]+\b', finnish_sent)
        
        # Filter out common articles from role names
        english_uppercase_words = [word for word in english_uppercase_words if word not in ["The", "An", "No", "There"]]
        finnish_uppercase_words = [word for word in finnish_uppercase_words if word not in ["The", "An", "No", "There"]]
        
        # New game-related keywords and their English/Finnish correspondences
        fullwords_to_fragments  = {
            "seating order": ["seating order", "istumajärjestys"],
            "outsider": ["outsider", "outsider"],
            "count": ["count", "määr"],
            "chosen": ["chose", "choose", "valin", "vali"],
            "modified": ["muokattu", "modified"],
            "poison": ["myrk", "poison"],
            "drunk": ["humal", "drunk" "juop"],
            "target": ["target", "kohd"],
            "evil": ["evil", "paha"],
            "good": ["good", "hyv"],
            "triggered": ["triggered", "lauka"],
            "kill": ["kill", "tappa"],
            "execute": ["execute", "teloitu"],
            "vote": ["vote", "äänest"],
            "death": ["death", "kuole"],
            "visit": ["visit", "vierail"],
            "ability": ["abilit", "voim"],
            "protect": ["protec", "suojel"],
            "false": ["false", "väär"],
            "night": ["night", "yö"],
            "day": ["day", "päiv"],
            "power": ["power", "voim"],
            "change": ["change", "vaiht", "muutt"],
            "minion": ["minion", "Minion","apula"]
        }


        # Extract keywords from the sentences
        extracted_keywords = []
        for fullword, fragments in fullwords_to_fragments.items():
            for fragment in fragments:
                if fragment in english_sent.lower() or fragment in finnish_sent.lower():
                    extracted_keywords.append(fullword)
                    break  # exit the inner loop once the keyword is found

        return {
            "english_roles": english_uppercase_words,
            "english_keywords": extracted_keywords
        }

    def create_prompt(self, english_row, finnish_row) -> str:
        info = self.extract_info(english_row, finnish_row)
        
        # Combine roles and keywords, then create a unique set to remove duplicates
        combined_set = set(info['english_roles'] + info['english_keywords'])
        
        # Convert the set back into a comma-separated string
        combined_str = ', '.join(combined_set)

        return combined_str
        
    def split_data(self):
        train, temp = train_test_split(self.df, random_state=42)
        valid, test = train_test_split(temp, train_size=0.5, random_state=42)
        return train, valid, test

    def get_dataset_dict(self):
        train, valid, test = self.split_data()
        self.train_dataset = self.create_dataset(train)
        self.valid_dataset = self.create_dataset(valid)
        self.test_dataset = self.create_dataset(test)

        dataset_dict = DatasetDict({
            'train': self.train_dataset,
            'validation': self.valid_dataset,
            'evaluation': self.test_dataset,
        })

        return dataset_dict
        
if __name__ == "__main__":
    # Create an instance of BOTC_Dataloader
    lang = "finnish"
    dataloader = BOTC_Dataloader(lang)  # Assuming the dataset contains both English and Finnish columns

    # Get the train dataloader
    train_dataloader, _, _ = dataloader.get_dataloaders()

    # The following loop processes each batch
    for batch in train_dataloader:
        inputs = batch['input_ids']

        # Decode the sentences
        sentences = dataloader.tokenizer.batch_decode(inputs, skip_special_tokens=True)

        for sentence in sentences:
            info = dataloader.extract_info(sentence, sentence)
            roles = ', '.join(info['english_roles'])
            keywords = ', '.join(info['english_keywords'])

            # Check if keywords list is empty and print the sentence if so
            if keywords:
                print(f"{keywords}, {roles}: {sentence}\n")

