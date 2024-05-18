#importing all the necessary libraries
import os
import math
import time
import sys
from collections import Counter
import matplotlib.pyplot as plt

def count_frequencies(tokens):
    return Counter(tokens)                                                                              #Counting the frequency of each token in a document.

def calculate_tf_idf(document_tokens, df, num_docs):                                                    #Calculating TF-IDF values for each document and its terms.
    tf_idf_results = {}                                                                                 #dictionary to store the tf-idf scores of each document.
    for filename, tokens in document_tokens.items():
        doc_tf_idf = {}                                                                                 #dictionary to store each term tf-idf score in current document
        frequencies = count_frequencies(tokens)                                                         #calculating freq of each term
        doc_length = len(tokens)                                                                        #calculating document size
        for term, freq in frequencies.items():
            idf = math.log((num_docs) / (df.get(term, 0)) )                                             #calculating idf for each term
            doc_tf_idf[term] = (freq / doc_length) * idf                                                #calculating tf-idf score for each term
        
        
        norm_factor = math.sqrt(sum(value ** 2 for value in doc_tf_idf.values()))                       # Normalize the TF-IDF scores using square root of sum of squares
        for term in doc_tf_idf:
            doc_tf_idf[term] /= norm_factor                                                             #normalizing each term tf-idf score

        tf_idf_results[filename] = doc_tf_idf                                                           #Storing normalized tf-idf scores of each file
    return tf_idf_results

def calculate_bm25(document_tokens, df, num_docs, doc_lengths, avg_doc_length, k1=1.2, b=0.75):         #Calculating BM25 for each document and its terms.
    bm25_results = {}                                                                                   #dictionary to store the bm25 scores of each document.
    for filename, tokens in document_tokens.items():
        doc_length = doc_lengths[filename]                                                              #getting individual document length
        bm25_scores = {}                                                                                #dictionary to store each term bm25 score in current document
        frequencies = count_frequencies(tokens)                                                         #calculating freq of each term
        for term, freq in frequencies.items():
            idf = math.log((num_docs) / (df.get(term, 0)))                                              #calculating idf for each term
            tf = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (doc_length / avg_doc_length)))          #calculating tf for each term
            bm25_scores[term] = idf * tf                                                                #calculating bm25 score for each term.
        bm25_results[filename] = bm25_scores                                                            #storing all term weights for individual file.
    return bm25_results

def write_output_files(tf_idf_results, bm25_results, output_directory):                                 #Writing TF-IDF and BM25 values side by side to output files, using the original input filenames.
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)                                                                   #Creating the output directory.
    for filename in tf_idf_results:                                                                     #Iterating over the filenames in the TF-IDF results.
        output_filename = f"{os.path.splitext(filename)[0]}.wts"                                        #Generating the output filename by removing original extension and adding .wts.
        output_path = os.path.join(output_directory, output_filename)
        with open(output_path, 'w', encoding='utf-8') as file:                                          #Opening the output file in write mode.
            for term in tf_idf_results[filename]:
                tf_idf_weight = tf_idf_results[filename].get(term, 0)                                   #Getting the TF-IDF weight for the terms, default to 0 if not found.
                bm25_weight = bm25_results[filename].get(term, 0)                                       #Getting the BM25 weight for the terms, default to 0 if not found.
                file.write(f"{term}: TF-IDF={tf_idf_weight}, BM25={bm25_weight}\n")

def process_documents(input_directory, output_directory, stopwords, document_counts):                   #Function to process documents, calculate TF-IDF/BM25 values, and write to output files.
    timings = []                                                                                        #list to store the processing times for each document
    for count in document_counts:
        start_time = time.time()                                                                        #Record the start time
        
        document_tokens, doc_lengths = {}, {}
        filenames = sorted(os.listdir(input_directory))[:count]                                         #retrieving the files.
        for filename in filenames:
            with open(os.path.join(input_directory, filename), 'r', encoding='utf-8', errors='replace') as file:
                text = file.read()                                                                      #reading the file
                tokens= text.split()                                                                    #getting the tokens in the file.
                
                document_tokens[filename] = [token for token in tokens if token not in stopwords and len(token) > 1]    #preprocessing the tokens as required.
                doc_lengths[filename] = len(document_tokens[filename])                                  #storing the document length by considering total count of tokens in each document.

        num_docs = len(filenames)                                                                       #storing total number of documents in the corpus
        all_tokens_flat = [token for tokens in document_tokens.values() for token in tokens]            #list containing all tokens from all lists in the dictionary document_tokens
        df = count_frequencies(all_tokens_flat)                                                         #Counting the document frequency of each token.
        avg_doc_length = sum(doc_lengths.values()) / num_docs                                   
        
        tf_idf_results = calculate_tf_idf(document_tokens, df, num_docs)                                #Function call to tf-idf
        bm25_results = calculate_bm25(document_tokens, df, num_docs, doc_lengths, avg_doc_length)       #Function call to bm25
        
        write_output_files(tf_idf_results, bm25_results, output_directory)                              #Function call to write the results into each output file.
        
        end_time = time.time()                                                                          #Record the end time
        timings.append(end_time - start_time)                                                           #Storing the processing time for the current document count.


    return document_counts, timings

def plot_timing_results(document_counts, timings):                                                      #Plot the timing results of document processing.
    plt.plot(document_counts, timings, marker='o')
    plt.xlabel('Number of Documents')
    plt.ylabel('Time (seconds)')
    plt.title('Processing Time vs. Number of Documents')
    plt.grid(True)
    plt.savefig('processing_time_vs_documents.png')
    plt.show()                                                                                          #Displaying the plot.

if __name__ == "__main__":
    input_directory = sys.argv[1]                                                                       #taking command-line arguments
    output_directory = sys.argv[2]
    stopwords = ['a', 'about', 'above', 'according', 'across', 'actually', 'adj', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', "aren't", 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'began', 'begin', 'beginning', 'behind', 'being', 'beings', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'big', 'billion', 'both', 'but', 'by', 'c', 'came', 'can', "can't", 'cannot', 'caption', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'co', 'come', 'could', "couldn't", 'd', 'did', "didn't", 'differ', 'different', 'differently', 'do', 'does', "doesn't", "don't", 'done', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere', 'end', 'ended', 'ending', 'ends', 'enough', 'etc', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'except', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'fifty', 'find', 'finds', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'important', 'in', 'inc', 'indeed', 'instead', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', 'j', 'just', 'k', 'l', 'large', 'largely', 'last', 'later', 'latest', 'latter', 'latterly', 'least', 'less', 'let', "let's", 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'ltd', 'm', 'made', 'make', 'makes', 'making', 'man', 'many', 'may', 'maybe', 'me', 'meantime', 'meanwhile', 'member', 'members', 'men', 'might', 'million', 'miss', 'more', 'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'namely', 'necessary', 'need', 'needed', 'needing', 'needs', 'neither', 'never', 'nevertheless', 'new', 'newer', 'newest', 'next', 'nine', 'ninety', 'no', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', "one's", 'only', 'onto', 'open', 'opened', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'overall', 'own', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'recent', 'recently', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'seven', 'seventy', 'several', 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'show', 'showed', 'showing', 'shows', 'sides', 'since', 'six', 'sixty', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'state', 'states', 'still', 'stop', 'such', 'sure', 't', 'take', 'taken', 'taking', 'ten', 'than', 'that', "that'll", "that's", "that've", 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', "there'd", "there'll", "there're", "there's", "there've", 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'thing', 'things', 'think', 'thinks', 'thirty', 'this', 'those', 'though', 'thought', 'thoughts', 'thousand', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'towards', 'trillion', 'turn', 'turned', 'turning', 'turns', 'twenty', 'two', 'u', 'under', 'unless', 'unlike', 'unlikely', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'using', 'v', 'very', 'via', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', "wasn't", 'way', 'ways', 'we', "we'd", "we'll", "we're", "we've", 'well', 'wells', 'were', "weren't", 'what', "what'll", "what's", "what've", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who'd", "who'll", "who's", 'whoever', 'whole', 'whom', 'whomever', 'whose', 'why', 'will', 'with', 'within', 'without', "won't", 'work', 'worked', 'working', 'works', 'would', "wouldn't", 'x', 'y', 'year', 'years', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'young', 'younger', 'youngest', 'your', 'yours', 'yourself', 'yourselves', 'z'] #list of stopwords
    document_counts = [10, 20, 40, 80, 100, 200, 300, 400, 504]
    
    start_total_time= time.time()
    counts, timings = process_documents(input_directory, output_directory, stopwords, document_counts)  #Function call to start processing and calculating the results.
    end_total_time= time.time()
    total_processing_time = end_total_time - start_total_time                                           #Calculating total processing time.
    plot_timing_results(counts, timings)


    print("Document Count | Timing (seconds)")
    print("---------------------------------")
    for count, timing in zip(counts, timings):
        print(f"{count:14} | {timing:.2f}")                                                             #Printing processing times.
    
    print(f"\nTotal Processing Time: {total_processing_time:.2f} seconds")
