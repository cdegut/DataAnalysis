from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import random

def pad_sequence(sequence, target_length):
    current_length = len(sequence)
    if current_length >= target_length:
        return sequence
    else:
        additional_residues = ''.join(random.choice('ATCG') for _ in range(target_length - current_length))
        return sequence + additional_residues

def process_fasta(input_file, output_file, target_length=300):
    new_records = []

    with open(input_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            padded_sequence = pad_sequence(str(record.seq), target_length)
            new_record = SeqRecord(Seq(padded_sequence), id=record.id, description=record.description)
            new_records.append(new_record)

    with open(output_file, "w") as output_handle:
        SeqIO.write(new_records, output_handle, "fasta")

if __name__ == "__main__":
    input_filename = "NTD.fasta"  # replace with your input file name
    output_filename = "output.fasta"  # replace with your desired output 
    process_fasta(input_filename, output_filename)
