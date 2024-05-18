import os
import math
from collections import Counter
import sys

def calculate_tf_idf(document_tokens, df, num_docs):
    tf_idf_results = {}
    for filename, tokens in document_tokens.items():
        doc_tf_idf = {}
        frequencies = Counter(tokens)
        doc_length = len(tokens)
        for term, freq in frequencies.items():
            idf = math.log(num_docs / (df.get(term, 0) + 1))
            doc_tf_idf[term] = (freq / doc_length) * idf
        # Normalize the TF-IDF scores
        norm_factor = math.sqrt(sum(value ** 2 for value in doc_tf_idf.values()))
        for term in doc_tf_idf:
            doc_tf_idf[term] /= norm_factor
        tf_idf_results[filename] = doc_tf_idf
    return tf_idf_results

def process_documents(input_directory, output_directory):
    document_tokens = {}
    doc_lengths = {}
    filenames = [filename for filename in sorted(os.listdir(input_directory)) if not filename.startswith('.')]
    num_docs = len(filenames)
    for filename in filenames:
        with open(os.path.join(input_directory, filename), 'r', encoding='utf-8', errors='replace') as file:
            text = file.read()
            tokens= text.split()
            document_tokens[filename] = tokens
            doc_lengths[filename] = len(tokens)

    all_tokens_flat = [token for tokens in document_tokens.values() for token in tokens]
    df = Counter(all_tokens_flat)
    tf_idf_results = calculate_tf_idf(document_tokens, df, num_docs)
    write_output_files(tf_idf_results, output_directory)

def write_output_files(tf_idf_results, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    postings = {}
    for filename, weights in tf_idf_results.items():
        for term, weight in weights.items():
            postings.setdefault(term, []).append((filename, weight))

    # Writing postings and dictionary files
    postings_file_path = os.path.join(output_directory, 'postings.txt')
    dictionary_file_path = os.path.join(output_directory, 'dictionary.txt')

    with open(postings_file_path, 'w', encoding='utf-8') as postings_file, \
         open(dictionary_file_path, 'w', encoding='utf-8') as dictionary_file:

        for term, docs in postings.items():
            # Write to dictionary
            dictionary_file.write(f"{term}\n{len(docs)}\n{postings_file.tell() + 1}\n")
            # Write to postings
            for doc_id, weight in docs:
                postings_file.write(f"{doc_id}, {weight:.6f}\n")

if __name__ == "__main__":
    input_directory = sys.argv[1]  # The first command line argument is the input directory
    output_directory = sys.argv[2]  # The second command line argument is the output directory
    process_documents(input_directory, output_directory)
