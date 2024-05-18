import os                                                                                                               #Importing necessary libraries.
import re
import math
import numpy as np
import time

def load_document_vectors(path):                                                                                        #Function to load data from files and create document vectors based only on TF-IDF weights.
    documents = {}
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            filepath = os.path.join(path, filename)
            with open(filepath, 'r') as file:
                doc_id = filename.split('.')[0]
                documents[doc_id] = {}
                for line in file:
                    match = re.search(r"(\w+): TF-IDF=([\d.-]+)", line)                                                 #Using regular expression to extract tf-idf values.
                    if match:
                        term = match.group(1)
                        tfidf = float(match.group(2))
                        documents[doc_id][term] = tfidf
    return documents

def cosine_similarity(doc1, doc2):                                                                                      #Function to calculate cosine similarity between two documents.
    intersection = set(doc1.keys()) & set(doc2.keys())                                                                  #Calculating the intersection of terms present in both documents.
    num = sum([doc1[x] * doc2[x] for x in intersection])                                                                #Computing the the dot product of the vectors
    sum1 = sum([doc1[x]**2 for x in doc1.keys()])                                                                       #Computing the sum of squares of vector componentsfor each document.
    sum2 = sum([doc2[x]**2 for x in doc2.keys()])
    den = math.sqrt(sum1) * math.sqrt(sum2)
    if not den:
        return 0.0
    else:
        return float(num) / den

def build_similarity_matrix(documents):                                                                                 #Building the similarity matrix.
    doc_ids = list(documents.keys())
    size = len(doc_ids)                                                                                                 #Determining the total number of documents to define the dimensions of the similarity matrix.
    sim_matrix = np.zeros((size, size))                                                                                 #Initializing a square matrix of dimensions size x size, and filling with zeros.
    for i in range(size):                                                                                               #Iterating over each pair of documents to calculate the cosine similarity between them.
        for j in range(i, size):
            sim = cosine_similarity(documents[doc_ids[i]], documents[doc_ids[j]])
            sim_matrix[i][j] = sim                                                                                      #Assigning the similarity score to both (i, j) and (j, i) since the matrix is symmetric.
            sim_matrix[j][i] = sim  
    return sim_matrix, doc_ids

def hierarchical_clustering(sim_matrix, doc_ids, threshold=0.4):
    header = '\t' + '\t'.join(doc_ids)  
    np.savetxt('similarity_matrix2.txt', sim_matrix, fmt='%.4f', delimiter='\t', header=header, comments='')            #Saving similarity matrix.

    clusters = {i: [doc_ids[i]] for i in range(len(doc_ids))}
    active_indices = set(clusters.keys())                                                                               #Using a set to keep track of removed clusters.
    output_count = 0                                                                                                    #Counter for the number of lines printed.

    while len(active_indices) > 1:                                                                                      
        max_sim = -1                                                                                                    #Initialize max_sim to -1 to find the maximum similarity in the current iteration.
        pair_to_merge = None
        for i in list(active_indices):                                                                                  #Iterating over all active clusters using two nested loops to examine every possible pair of clusters.
            for j in list(active_indices):
                if i != j and sim_matrix[i][j] > max_sim:                                                               
                    max_sim = sim_matrix[i][j]
                    pair_to_merge = (i, j)

        if max_sim < threshold or output_count >= 100:                                                                 #If no pairs above the threshold, stop clustering.
            break

        cluster1, cluster2 = pair_to_merge                                                                             #Merging the two clusters.
        if output_count < 100:
            print(f"Merging clusters {clusters[cluster1]} and {clusters[cluster2]} with similarity {max_sim}")
            output_count += 1

        clusters[cluster1].extend(clusters[cluster2])                                                                  #Extending the first cluster with the second and deleting the second.
        del clusters[cluster2]                                                                                         #Removing the entry for the second cluster from the clusters dictionary.
        active_indices.remove(cluster2)                                                                                #Removing the index of the second cluster from the set of active_indices.

        for k in active_indices:                                                                                       #Updating the similarity matrix for the new cluster.
            if k != cluster1:
                merged_sim = np.mean([sim_matrix[cluster1][k], sim_matrix[cluster2][k]])                               #Using group average link method to update the similarity.
                sim_matrix[cluster1][k] = merged_sim
                sim_matrix[k][cluster1] = merged_sim
        sim_matrix[cluster2, :] = -1                                                                                   #Ensuring cluster2 entries are not used further.
        sim_matrix[:, cluster2] = -1

def find_similar_and_dissimilar_pairs(sim_matrix, doc_ids):                                                            #Function to find the most similar and most dissimilar pairs of documents.
    max_similarity = 0
    min_similarity = float('inf')
    most_similar_pair = None
    most_dissimilar_pair = None
    size = len(sim_matrix)

    for i in range(size):
        for j in range(i + 1, size):                                                                                   #Ensuring no self-comparison.
            if sim_matrix[i][j] > max_similarity:
                max_similarity = sim_matrix[i][j]
                most_similar_pair = (doc_ids[i], doc_ids[j])

            if 0 < sim_matrix[i][j] < min_similarity:                                                                  #Checking for dissimilarity.
                min_similarity = sim_matrix[i][j]
                most_dissimilar_pair = (doc_ids[i], doc_ids[j])
    return most_similar_pair, max_similarity, most_dissimilar_pair, min_similarity

def compute_centroid(documents):                                                                                       #Function to compute the centroid.
    centroid = {}
    for doc_id, weights in documents.items():
        for term, weight in weights.items():
            if term in centroid:
                centroid[term] += weight
            else:
                centroid[term] = weight
    
    num_docs = len(documents)                                                                                         #Average the sums to get the centroid.
    for term in centroid:
        centroid[term] /= num_docs
    return centroid

def find_closest_to_centroid(documents, centroid):                                                                   #Function to find the document closest to centroid.
    max_similarity = 0
    closest_doc = None
    for doc_id, doc_vector in documents.items():
        similarity = cosine_similarity(doc_vector, centroid)                                                         #Calculating cosine similarity.
        if similarity > max_similarity:
            max_similarity = similarity
            closest_doc = doc_id
    return closest_doc, max_similarity

if __name__ == "__main__":
    documents = load_document_vectors("/Users/saitejachalla/Desktop/IR/output3")                                     #Calling the function to load the document vectors from the specified directory.
    start_time= time.time()
    sim_matrix, doc_ids = build_similarity_matrix(documents)                                                         #Calling the function to build similarity matrix.
    
    hierarchical_clustering(sim_matrix, doc_ids)                                                                     #Calling the hierarchical clustering function.
    end_time= time.time()
    print("-------------------------------------------------------------------")
    print(f"Total execution time for clustering {end_time - start_time} seconds")                                    #Printing total execution time.
    print("-------------------------------------------------------------------")
    most_similar_pair, max_sim, most_dissimilar_pair, min_sim = find_similar_and_dissimilar_pairs(sim_matrix, doc_ids)
    print(f"Most similar pair: {most_similar_pair} with similarity {max_sim}")                                       #Printing the most similar and dissimilar pairs.
    print(f"Most dissimilar pair: {most_dissimilar_pair} with similarity {min_sim}")

    centroid = compute_centroid(documents)
    closest_doc, closest_sim = find_closest_to_centroid(documents, centroid)
    print(f"Document closest to centroid: {closest_doc} with similarity {closest_sim}")                              #Printing the document closest to centroid.