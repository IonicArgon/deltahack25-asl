from datasets import Dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import os
import torch
from PIL import Image
import cv2

data_path = 'data/self_curated_2/train'

data = []

for label in os.listdir(data_path):
    folder_path = os.path.join(data_path, label)
    if os.path.isdir(folder_path):
        for image in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image)
            data.append({'image': image_path, 'label': label})

dataset = Dataset.from_list(data)
labels = dataset.unique('label')

label2id = {label: id for id, label in enumerate(labels)}
id2label = {id: label for id, label in enumerate(labels)}

dataset = dataset.add_column('label_id', [label2id[label] for label in dataset['label']])

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

def preprocess_function(examples):
    images = [Image.open(image) for image in examples['image']]
    inputs = feature_extractor(images=images, return_tensors='pt')
    inputs['label'] = torch.tensor(examples['label_id'], dtype=torch.int)
    return inputs

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.train_test_split(test_size=0.1)

print(encoded_dataset)

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224', 
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir='./results/ds2',          
    num_train_epochs=3,              
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',
    report_to='none',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=encoded_dataset['train'],         
    eval_dataset=encoded_dataset['test']            
)

if torch.cuda.is_available():
    model.cuda()

trainer.train()
