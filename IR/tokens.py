#importing required libraries
import os                                                   # For file operations
import sys                                                  # For taking command-line arguments
from bs4 import BeautifulSoup                               # For parsing HTML documents
from nltk.tokenize import word_tokenize                     # For tokenizing text into words
from collections import Counter                             # For tokens count/frequency
import re                                                   # For regular expressions use
import time                                                 # For calculating elapsed time and CPU time
import matplotlib.pyplot as plt                             # For plotting graph
import pandas as pd                                         # For storing time data in data frames

plt.rcParams['font.family'] = 'sans-serif'                  # Setting font family

import nltk
nltk.download('punkt')

def parsing_html(file_path):                                  # Function for paring each HTML file and returning text content.
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:    # opening the file
        soup = BeautifulSoup(file, 'html.parser')           # parsing the opened file using beautifulsoap.
        text = soup.get_text()                              # getting the text and storing it in a variable.
        text = re.sub(r"\d+", '', text)                     # removing digits from the parsed_text.
        text = re.sub(r"'", '', text)                       # removing apostrophes from the parsed_text.
        text = re.sub(r"_", '', text)                       # removing underscores from the parsed_text.
        text = re.sub(r"-", '', text)                       # removing hyphens from the parsed_text.
        text = re.sub(r"\.", '', text)                      # Remove additional symbols: comma, period,
        text = re.sub(r"[,.?!()\[\]]", '', text)            # question mark, exclamation mark, parentheses, square brackets.
        text = re.sub(r"[^\w\s]", '', text)                 # Remove non-alphabetic characters except spaces

    return text                                             # returing the cleaned text


def text_conversion(text):                           # Function for tokenizing text and convert tokens to lowercase.
    tokens = word_tokenize(text)
    lower_tokens = [token.lower() for token in tokens]
    return lower_tokens

def process_document(file_path, output_dir):                # Function for processing a single document: parse, tokenize, lowercase, and write output.
    text = parsing_html(file_path)
    tokens = text_conversion(text)
    file_base_name = os.path.basename(file_path)                 # Storing the basename of output path for output file.
    output_file_name = os.path.splitext(file_base_name)[0] + '.txt' 
    output_file_path = os.path.join(output_dir, output_file_name)
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        outfile.write(' '.join(tokens))                     # Writing the tokenized text to the output file.
    return tokens

def first_last(file_path, n):                   # Functiong to store the first and last n lines to separate files.
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()                            # Reading the file lines.
    #retrieving the first and last 50 lines using slicing operator.
    first_n_lines = lines[:n]
    last_n_lines = lines[-n:]                               # spiltting the lines.

    filebase_name = os.path.splitext(os.path.basename(file_path))[0]
    first_n_file = f'{filebase_name}_first_{n}.txt'             # Storing the basename of the path for first and last 50 lines file.
    last_n_file = f'{filebase_name}_last_{n}.txt'

    with open(first_n_file, 'w', encoding='utf-8') as file:
        file.writelines(first_n_lines)                      # Writing the first 50 lines into a seperate file.

    with open(last_n_file, 'w', encoding='utf-8') as file:
        file.writelines(last_n_lines)                       # Writing the last 50 lines into a seperate file.



def process_documents(input_dir, output_dir):               # Process all documents in the input directory and generate frequency files.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_tokens = []
    elapsed_times = []
    cpu_times = []
    num_documents_processed = []
    time_for_100_docs = []

    
    file_names = os.listdir(input_dir)                      # Process each file in the input directory
    num_files = len(file_names)
    for i, file_name in enumerate(file_names, 1):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path):
            start_cpu_time = time.process_time()            # Record start CPU time
            start_elapsed_time = time.time()                # Record start elapsed time
            
            tokens = process_document(file_path, output_dir)
            all_tokens.extend(tokens)
            
            end_cpu_time = time.process_time()              # Record end CPU time
            end_elapsed_time = time.time()                  # Record end elapsed time
            
            elapsed_time = (end_elapsed_time - start_elapsed_time) * 1000
            cpu_time = (end_cpu_time - start_cpu_time) * 1000
            
            elapsed_times.append(elapsed_time)
            cpu_times.append(cpu_time)
            num_documents_processed.append(i)
            if i % 100 == 0:
                time_for_100_docs.append(sum(elapsed_times[-100:]))

    
    token_freq = Counter(all_tokens)                        # Calculate frequencies

    # Writing the two frequency files
    #1 Sorted by token:
    with open(os.path.join(output_dir, 'tokens_sorted_by_token.txt'), 'w', encoding='utf-8') as file:
        for token, freq in sorted(token_freq.items()):
            file.write(f"{token}: {freq}\n")


    #2 Sorted by frequency:
    with open(os.path.join(output_dir, 'tokens_sorted_by_frequency.txt'), 'w', encoding='utf-8') as file:
        for token, freq in token_freq.most_common():
            file.write(f"{token}: {freq}\n")



    # Writing first and last 50 lines for each file
    first_last('/Users/saitejachalla/Desktop/IR/output/tokens_sorted_by_frequency.txt',50)
    first_last('/Users/saitejachalla/Desktop/IR/output/tokens_sorted_by_token.txt',50)



    # Generate a table for time taken for every 100 documents processed.
    num_hundred_docs = len(time_for_100_docs)
    time_df = pd.DataFrame({'Documents Processed': range(100, num_hundred_docs * 100 + 1, 100),
                            'Elapsed Time (milliseconds)': time_for_100_docs,
                            'CPU Time (milliseconds)': cpu_times[:num_hundred_docs]})  # Limit cpu_times to match num_hundred_docs


 
    # Plot for Elapsed Time
    plt.figure(figsize=(10, 6)) 
    plt.plot(num_documents_processed, elapsed_times, label='Elapsed Time', color='blue')
    plt.xlabel('Set of Documents Processed')
    plt.ylabel('Time in milliseconds')
    plt.title('Elapsed Time vs Set of Documents Processed')
    plt.legend()
    plt.grid(True)
    plt.show()                                              # Displays the first plot

    # Plot for CPU Time
    plt.figure(figsize=(10, 6))  
    plt.plot(num_documents_processed, cpu_times, label='CPU Time', color='red')
    plt.xlabel('Set of Documents Processed')
    plt.ylabel('Time in milliseconds')
    plt.title('CPU Time vs Set of Documents Processed')
    plt.legend()
    plt.grid(True)
    plt.show()                                              # Displays the second plot

    
    total_time = (sum(elapsed_times)) * 1000                # Calculating the total time
    print(f"Total time taken: {total_time} milliseconds")

    return time_df


def print_statistics(time_df, total_time):
    #Print statistics including time taken for every 100 documents processed with CPU and elapsed time.
    print("Time taken for every 100 documents:")            # Print the time taken for every 100 documents.
    print(time_df)
    #print(f"Total time taken for processing all documents: {total_time} seconds")

if __name__ == "__main__":
    input_path = sys.argv[1]                             # taking command-line arguments
    output_path = sys.argv[2]
    
    time_df = process_documents(input_path, output_path)
    total_time = sum(time_df['Elapsed Time (milliseconds)'])
    print_statistics(time_df, total_time)                   # Calculate and print statistics.
    