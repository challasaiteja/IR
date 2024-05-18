#importing all the required libraries
import os
import math
from collections import Counter
import sys
import matplotlib.pyplot as plt
import time

def calculate_tf_idf(document_tokens, df, num_docs):                                                    #Calculating TF-IDF values for each document and its terms.
    tf_idf_results = {}                                                                                 #dictionary to store the tf-idf scores of each document.
    for filename, tokens in document_tokens.items():
        doc_tf_idf = {}                                                                                 #dictionary to store each term tf-idf score in current document
        frequencies = Counter(tokens)                                                                   #calculating freq of each term
        doc_length = len(tokens)                                                                        #calculating document size
        for term, freq in frequencies.items():
            idf = math.log((num_docs) / (df.get(term, 0) + 1))                                          #calculating idf for each term
            doc_tf_idf[term] = (freq / doc_length) * idf                                                #calculating tf-idf score for each term
        
        
        norm_factor = math.sqrt(sum(value ** 2 for value in doc_tf_idf.values()))                       #Normalize the TF-IDF scores using square root of sum of squares
        for term in doc_tf_idf:
            doc_tf_idf[term] /= norm_factor                                                             #normalizing each term tf-idf score

        tf_idf_results[filename] = doc_tf_idf                                                           #Storing normalized tf-idf scores of each file
    return tf_idf_results

def process_documents(input_directory, output_directory, stopwords, document_counts):
    indexing_times = []                                                                                 #list to store the processing times for each document                                                                                   
    for count in document_counts:
        start_time = time.time()                                                                        #record start time

        document_tokens, doc_lengths = {}, {}
        filenames = [filename for filename in sorted(os.listdir(input_directory))[:count] if not filename.startswith('.')]
        for filename in filenames:
            with open(os.path.join(input_directory, filename), 'r', encoding='utf-8', errors='replace') as file:
                text = file.read()                                                                      #reading the file
                tokens= text.split()                                                                    #getting the tokens in the file.
                
                document_tokens[filename] = [token for token in tokens if token not in stopwords and len(token) > 1]    #preprocessing the tokens as required.
                doc_lengths[filename] = len(document_tokens[filename])                                  #storing the document length by considering total count of tokens in each document.

        num_docs = len(filenames)                                                                       #Storing total number of documents in the corpus
        all_tokens_flat = [token for tokens in document_tokens.values() for token in tokens]            #List containing all tokens from all lists in the dictionary document_tokens
        df = Counter(all_tokens_flat)                                                                   #Counting the document frequency of each token.
        tf_idf_results = calculate_tf_idf(document_tokens, df, num_docs)                                #Calling function to calculate tf-idf and store the results.
        write_output_files(tf_idf_results, output_directory)                                            #Calling function to write results to outputfiles.

        indexing_time = time.time() - start_time                                                        #Record end time
        indexing_times.append(indexing_time)                                                            #storing processing time for current document count.

    return indexing_times

def plot_indexing_times(document_counts, indexing_times):                                               #plotting the graph between the document count and timings.
    plt.plot(document_counts, indexing_times, marker='o')
    plt.xlabel('Number of Documents')
    plt.ylabel('Indexing Time (seconds)')
    plt.title('Indexing Time vs. Number of Documents')
    plt.grid(True)
    plt.savefig('index_processing_times_vs_documents.png')
    plt.show()

def write_output_files(tf_idf_results, output_directory):                                               
    if not os.path.exists(output_directory):                                                            #Creating directory if necessary.
        os.makedirs(output_directory)

    postings = {}                                                                                       #Intializing a dictionary to store the postings lists for each term.
    for filename, weights in tf_idf_results.items():
        for term, weight in weights.items():
            postings.setdefault(term, []).append((filename, weight))                                    #For each term, we are appending a tuple of (filename, weight) to the postings list.('setdefault' is used to initialize the list for a term if it does not already exist.)

    postings_file_path = os.path.join(output_directory, 'postings2.txt')                                
    dictionary_file_path = os.path.join(output_directory, 'dictionary2.txt')                            

    with open(postings_file_path, 'w', encoding='utf-8') as postings_file, \
         open(dictionary_file_path, 'w', encoding='utf-8') as dictionary_file:                          #Opening the both files to write.

        for term, docs in postings.items():
            dictionary_file.write(f"{term}\n{len(docs)}\n{postings_file.tell() + 1}\n")                 #Writing the term, number of documents containing the term, and the current byte offset in the postings file to the dictionary file.
            
            for doc_id, weight in docs:
                postings_file.write(f"{doc_id.split('.')[0]}, {weight:.6f}\n")                          #For each document that contains the current term, write the document ID (stripped of '.txt') and the term's weight in that document to the postings file.

if __name__ == "__main__":
    input_directory = sys.argv[1]                                                                       #Taking the command line arguments.
    output_directory = sys.argv[2]  
    stopwords = ['a', 'about', 'above', 'according', 'across', 'actually', 'adj', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', "aren't", 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'began', 'begin', 'beginning', 'behind', 'being', 'beings', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'big', 'billion', 'both', 'but', 'by', 'c', 'came', 'can', "can't", 'cannot', 'caption', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'co', 'come', 'could', "couldn't", 'd', 'did', "didn't", 'differ', 'different', 'differently', 'do', 'does', "doesn't", "don't", 'done', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere', 'end', 'ended', 'ending', 'ends', 'enough', 'etc', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'except', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'fifty', 'find', 'finds', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'important', 'in', 'inc', 'indeed', 'instead', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', 'j', 'just', 'k', 'l', 'large', 'largely', 'last', 'later', 'latest', 'latter', 'latterly', 'least', 'less', 'let', "let's", 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'ltd', 'm', 'made', 'make', 'makes', 'making', 'man', 'many', 'may', 'maybe', 'me', 'meantime', 'meanwhile', 'member', 'members', 'men', 'might', 'million', 'miss', 'more', 'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'namely', 'necessary', 'need', 'needed', 'needing', 'needs', 'neither', 'never', 'nevertheless', 'new', 'newer', 'newest', 'next', 'nine', 'ninety', 'no', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', "one's", 'only', 'onto', 'open', 'opened', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'overall', 'own', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'recent', 'recently', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'seven', 'seventy', 'several', 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'show', 'showed', 'showing', 'shows', 'sides', 'since', 'six', 'sixty', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'state', 'states', 'still', 'stop', 'such', 'sure', 't', 'take', 'taken', 'taking', 'ten', 'than', 'that', "that'll", "that's", "that've", 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', "there'd", "there'll", "there're", "there's", "there've", 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'thing', 'things', 'think', 'thinks', 'thirty', 'this', 'those', 'though', 'thought', 'thoughts', 'thousand', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'towards', 'trillion', 'turn', 'turned', 'turning', 'turns', 'twenty', 'two', 'u', 'under', 'unless', 'unlike', 'unlikely', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'using', 'v', 'very', 'via', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', "wasn't", 'way', 'ways', 'we', "we'd", "we'll", "we're", "we've", 'well', 'wells', 'were', "weren't", 'what', "what'll", "what's", "what've", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who'd", "who'll", "who's", 'whoever', 'whole', 'whom', 'whomever', 'whose', 'why', 'will', 'with', 'within', 'without', "won't", 'work', 'worked', 'working', 'works', 'would', "wouldn't", 'x', 'y', 'year', 'years', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'young', 'younger', 'youngest', 'your', 'yours', 'yourself', 'yourselves', 'z'] #list of stopwords
    document_counts = [10, 20, 40, 80, 100, 200, 300, 400, 504]
    
    starting= time.time()
    indexing_times = process_documents(input_directory, output_directory, stopwords, document_counts)
    ending= time.time()
    total_processing_time= ending - starting

    plot_indexing_times(document_counts, indexing_times)
    print("Document Count | Timing (seconds)")
    print("---------------------------------")
    for document_counts, indexing_times in zip(document_counts, indexing_times):
        print(f"{document_counts:14} | {indexing_times:.2f}")                                             #Printing processing times.
    
    print(f"\nTotal Processing Time: {total_processing_time:.2f} seconds")