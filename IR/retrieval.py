import sys                                                                                                        #Importing the necessary libraries.

def preprocess(phrase, stopwords):
    words = [word.lower() for word in phrase.split() if word.lower() not in stopwords]                            #Split the phrase into words and filter out stopwords.
    return ' '.join(words)                                                                                        #Joining words and returing it.

def load_dictionary(file_path):
    dictionary = {}                                                                                               #Initializing a dictionary to store the index.                                                                                                                                                                        
    with open(file_path, 'r') as file:
        while True:                                                                                               #Reading the lines in the dictionary file.
            word = file.readline().strip()
            if not word:
                break
            doc_count = int(file.readline().strip())
            position = int(file.readline().strip())
            dictionary[word] = {'doc_count': doc_count, 'position': position}                                     #Adding the word to the dictionary with its document count and position in postings as a sub-dictionary.
    return dictionary

def load_postings(file_path, position, doc_count):
    postings = {}                                                                                                 #Initializing a dictionary to store the postings list.    
    with open(file_path, 'r') as file:
        file.seek(position)
        #print(f"Seeking to position: {position}")
        for _ in range(doc_count):                                                                                #Iterating over the number of documents that contain the term.
            line = file.readline().strip()                                                                        #Reading each line corresponding to a posting.
            #print(f"Read line: '{line}'")
            if line:
                doc_id, weight = line.split(', ')                                                                 #Retrieving document id and weight.
                postings[doc_id.zfill(3)] = float(weight)                                                         #Adding the document id and weight to the postings dictionary.
                #print(f"Loaded posting: {doc_id} with weight {weight}")
    return postings

def rank_documents(query_terms, dictionary, postings_file):
    document_scores = {}                                                                                           #Initializing a dictionary to store the cumulative scores for each document.
    for phrase, weight in query_terms:                                                                             #Iterating over each term and its associated weight in the query.
        words = phrase.split()
       
        for word in words:                                                                                         #If the query contains multiple words, processing them individually
            term_info = dictionary.get(word)                                                                       #Retrieving the dictionary entry for the word.
            if term_info:
                term_postings = load_postings(postings_file, term_info['position'], term_info['doc_count'])        #Loading the postings list for the word.
                for doc_id, term_weight in term_postings.items():                                                  #Iterating over each document ID and term weight in the postings list.
                    if doc_id not in document_scores:
                        document_scores[doc_id] = 0.0                                                              #If the document ID isn't already in the document_scores dictionary, add it with a starting score of 0.0.
                    document_scores[doc_id] += weight * term_weight                                                #Adding the product of the query term's weight and the term's weight in the document into the dictionary.
    return document_scores

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide prompt as python filename.py 'query_terms' weight ['query_terms' weight ...]")
        sys.exit(1)

    stopwords = ['a', 'about', 'above', 'according', 'across', 'actually', 'adj', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', "aren't", 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'began', 'begin', 'beginning', 'behind', 'being', 'beings', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'big', 'billion', 'both', 'but', 'by', 'c', 'came', 'can', "can't", 'cannot', 'caption', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'co', 'come', 'could', "couldn't", 'd', 'did', "didn't", 'differ', 'different', 'differently', 'do', 'does', "doesn't", "don't", 'done', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere', 'end', 'ended', 'ending', 'ends', 'enough', 'etc', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'except', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'fifty', 'find', 'finds', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'important', 'in', 'inc', 'indeed', 'instead', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', 'j', 'just', 'k', 'l', 'large', 'largely', 'last', 'later', 'latest', 'latter', 'latterly', 'least', 'less', 'let', "let's", 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'ltd', 'm', 'made', 'make', 'makes', 'making', 'man', 'many', 'may', 'maybe', 'me', 'meantime', 'meanwhile', 'member', 'members', 'men', 'might', 'million', 'miss', 'more', 'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'namely', 'necessary', 'need', 'needed', 'needing', 'needs', 'neither', 'never', 'nevertheless', 'new', 'newer', 'newest', 'next', 'nine', 'ninety', 'no', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', "one's", 'only', 'onto', 'open', 'opened', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'overall', 'own', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'recent', 'recently', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'seven', 'seventy', 'several', 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'show', 'showed', 'showing', 'shows', 'sides', 'since', 'six', 'sixty', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'state', 'states', 'still', 'stop', 'such', 'sure', 't', 'take', 'taken', 'taking', 'ten', 'than', 'that', "that'll", "that's", "that've", 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', "there'd", "there'll", "there're", "there's", "there've", 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'thing', 'things', 'think', 'thinks', 'thirty', 'this', 'those', 'though', 'thought', 'thoughts', 'thousand', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'towards', 'trillion', 'turn', 'turned', 'turning', 'turns', 'twenty', 'two', 'u', 'under', 'unless', 'unlike', 'unlikely', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'using', 'v', 'very', 'via', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', "wasn't", 'way', 'ways', 'we', "we'd", "we'll", "we're", "we've", 'well', 'wells', 'were', "weren't", 'what', "what'll", "what's", "what've", 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who'd", "who'll", "who's", 'whoever', 'whole', 'whom', 'whomever', 'whose', 'why', 'will', 'with', 'within', 'without', "won't", 'work', 'worked', 'working', 'works', 'would', "wouldn't", 'x', 'y', 'year', 'years', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'young', 'younger', 'youngest', 'your', 'yours', 'yourself', 'yourselves', 'z'] #list of stopwords

    args = iter(sys.argv[1:])                                                                                       #Processing the command line arguments into a list of tuples (phrase, weight).
    queries = [(preprocess(phrase, stopwords), float(weight)) for phrase, weight in zip(args, args)]                #Expecting the input format: "phrase" weight "phrase" weight
    dictionary = load_dictionary('/Users/saitejachalla/Desktop/IR/dictionary2.txt')                                 #calling the function to load dictionary file.
    document_scores = rank_documents(queries, dictionary, '/Users/saitejachalla/Desktop/IR/postings2.txt')          #calling the rank_documents functions based on the query.
    document_scores = {doc_id: score for doc_id, score in document_scores.items() if score > 0}                     #Filtering out the documents with a score of zero.
    sorted_docs = sorted(document_scores.items(), key=lambda item: item[1], reverse=True)                           #Sorting the documents by their scores in descending order.
    
    if sorted_docs:                                                                                                 #Displaying the top 10 documents with non-zero scores
        print("The ten top-ranking filenames:")
        for doc_id, score in sorted_docs[:10]:
            print(f" {doc_id}.html: {score:.9f}")
    else:
        print("There are no files with these key words in the given corpus.")