from gc import callbacks
from pyparsing import col
import tensorflow as tf
import datasets
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel
from transformers import AutoTokenizer,DataCollatorWithPadding,create_optimizer

import numpy

from transformers import TFAutoModelForSequenceClassification
print(tf.__version__) # print the current version
#print(tf.config.list_physical_devices()) # print available physical devices

# Loading Local Dataset

#dataset = datasets.load_dataset('csv', data_files='all_articles_cleaned.csv')

df = pd.read_csv('all_articles_cleaned.csv')

df = df.drop(columns=['URL', 'Title'])
df= df.drop(df.columns[0], axis=1)

df['Alignment'] = df['Alignment'].replace({
    'right': 0,
    'right-center': 1,
    'center': 2,
    'left-center': 3,
    'left': 4
})
df = df.rename(columns={"Alignment":"labels", "Text": "text"})

print(df)
class_names = ['0' , '1', '2', '3', '4']
features = Features({"text": Value("string"), "labels": ClassLabel(num_classes=5, names=class_names)})


dataset = Dataset.from_pandas(df, features=features)
#dataset = dataset.class_encode_column("labels")
train_testvalid = dataset.train_test_split(test_size=0.10)
'''
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = datasets.DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

'''
print(train_testvalid)

from transformers import BertTokenizerFast

#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

def preprocess_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True)

pre_tokenizer_columns = set(train_testvalid["train"].features)

#cols = train_testvalid["train"].column_names
#cols.remove("labels")

tokenized_data = train_testvalid.map(preprocess_function, batched=True)
tokenizer_columns = list(set(tokenized_data["train"].features) - pre_tokenizer_columns)

print("Columns added by tokenizer:", tokenizer_columns)
print(tokenized_data["train"].features)

print(tokenized_data)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")


tf_train_set = tokenized_data["train"].to_tf_dataset(
    columns=tokenizer_columns,
    label_cols="labels",
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_set = tokenized_data["test"].to_tf_dataset(
    columns=tokenizer_columns,
    label_cols="labels",
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

print(tf_train_set)
print(tf_validation_set)

batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_data["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)


model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
#model.compile(optimizer=optimizer, metrics=['accuracy'])
#model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=1)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./text_classification_model_save/logs")




model.fit(tf_train_set, validation_data=tf_validation_set, epochs=4, callbacks=[tensorboard_callback])


model.save_pretrained("./bert-text-classification-news")
tokenizer.save_pretrained("./bert-text-classification-news")

tokenizer.push_to_hub("bert-text-classification-news")
model.push_to_hub("bert-text-classification-news")



