
# import
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.cluster import KMeans

# BERT model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Read in
file_path = "scatch_4.txt"

with open(file_path, "r", encoding="utf-8") as file:
    incidents = file.readlines()

# tokenize section
tokenized_descriptions = [tokenizer.encode(incident, add_special_tokens=True, max_length=512, truncation=True) for incident in incidents]

# BERT embeddings output
with torch.no_grad():
    embeddings = model(torch.tensor(tokenized_descriptions))

#k-means for clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(embeddings)


for i, incident in enumerate(incidents):
    print(f"Incident {i + 1}: Cluster {clusters[i]} - {incident}")