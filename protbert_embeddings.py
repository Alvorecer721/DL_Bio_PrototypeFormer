import os
from Bio import SeqIO
from transformers import BertModel, BertTokenizer
import torch
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

'''original_pt_folder = "./fewshotbench/data/swissprot/embeds/"  
subset_fasta_path = "subset.fasta"  

pt_files = [file for file in os.listdir(original_pt_folder) if file.endswith(".pt")]

protein_ids = [file.split('.')[0] for file in pt_files]
'''
'''
    Because the original embedding did not use all the proteins in the dataset, 
    we use the same protein id as the original embedding to generate the new embedding using protbert.
    The used protein sequences are stored in the file subset_fasta.fasta.
'''

'''with open("./fewshotbench/data/swissprot/uniprot_sprot.fasta", "r") as original_fasta, open(subset_fasta_path, "w") as subset_fasta:
    records = [record for record in SeqIO.parse(original_fasta, "fasta") if record.id.split("|")[1] in protein_ids]
    SeqIO.write(records, subset_fasta, "fasta")
'''

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert").to(device)

save_path = "./fewshotbench/data/swissprot/protbert_emb"
os.makedirs(save_path, exist_ok=True)


with open("subset.fasta", "r") as fasta_file:
    line = fasta_file.readline().strip()
    while line != '':
        if line.startswith(">"):  
            protein_id = line.split("|")[1]
            print(protein_id)
            sequence_line = ''
            line = fasta_file.readline().strip()
            while line != '' and not line.startswith(">"):
                sequence_line += line
                line = fasta_file.readline().strip()
            
            sequence = re.sub(r"[UZOB]", "X", sequence_line)

            encoded_input = tokenizer(sequence, return_tensors="pt").to(device)
            output = model(**encoded_input)

            embedding = output["last_hidden_state"].mean(dim=1).cpu().detach()
            print(embedding)

            torch.save({"embedding": embedding}, f"{save_path}/{protein_id}.pt")
            