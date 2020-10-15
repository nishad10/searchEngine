#######################
# Name: Nishad Aherrao
# ID:   1001351291
#######################

import os
import math
import operator
from collections import Counter
from collections import OrderedDict
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Your dir path that contains files to compare query with
corpusroot = './presidential_debates'
lis = stopwords.words('english')
stopwordSorted = sorted(lis)

# Empty Containers
wordListAll = []  # All tokens
words = {}  # Dict of all tokens as key and value is the document frequency
querytf = {}  # The weights for query
# Dict with final answer of docs their similarity scores and if pure or not (true, false)
docSimilarity = {}
filenames = []  # Filenames array
tf_idf = {}  # Final dict of tf_idf scores for given [document, token]  = score
idfList = {}  # Idf scores for tokens


def tokenize(doc):  # Tokenize the document
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(doc)
    return tokens


def preprocessing(doc, stopwords):  # A function that does preprocessing on a document
    doc = doc.lower()
    tokens = tokenize(doc)
    tokensNoStopWords = removeStopwords(tokens, stopwords)
    stemmer = PorterStemmer()
    tokenStem = [stemmer.stem(word) for word in tokensNoStopWords]
    return tokenStem


# Used in preprocessing, this removes stopwords given list of tokens
def removeStopwords(tokens, stopwords):
    sanitized = []
    for word in tokens:
        if word not in stopwords:
            sanitized.append(word)
    return sanitized


def unique(list1):  # gives us a unique list using the set hack
    list_set = set(list1)
    unique_list = (list(list_set))
    return unique_list


def document_frequency(word):  # gives the document frequency of a token
    c = 0
    try:
        c = words[word]
    except:
        c = 0
    return c


def getidf(token):  # Used to answer our questions for grading, gives idf values from dictionary
    return idfList[token]


# Used to answer our questions gives final normalized weights for a token given a filename
def getweight(filename, token):
    # get the doc number based on the file name
    doc = filenames.index(filename)
    try:
        return tf_idf[doc, token]
    except:
        return 0


def query(queryString):  # Get the most similar document and its similarity based on query
    # Get query tokens
    queryTokens = preprocessing(queryString, stopwordSorted)
    # The list of top 10 documents based on their tf-idf weights of format {token1 : [...], token2: [...]}
    top10List = {}
    # A list of documents and their weights for a token of format [{token1docNumber1: tf-idfweight1, ...}, {token1docNumber2: tf-idfweight2, ...}]
    allElementList = []
    for queryToken in queryTokens:
        top10counter = 0
        for k, v in tf_idf.items():
            if(k[1] == queryToken and top10counter < 10):
                top10counter += 1
                try:
                    top10List[queryToken][k[0]] = v
                except:
                    top10List[queryToken] = {k[0]: v}
    if(not top10List):
        return('None', 0)
    for k, elementList in top10List.items():
        allElementList.append(elementList)

    # Again use set to get unique list of documents
    # This is basically a unique list of all documents from all top 10 list of all tokens
    common_documents = set(allElementList[0].keys())

    totalquerytf = 0  # Used for query tf normalizing (Magnitude)

    # For every unique token from query tokens calculate its tf-idf weights and normalize
    for queryToken in unique(queryTokens):
        counter = Counter(queryTokens)
        tf = counter[queryToken]
        tf = (1 + math.log(tf, 10)) if tf > 0 else 0
        querytf[queryToken] = tf
        totalquerytf += (querytf[queryToken] *
                         querytf[queryToken])  # magnitude

    # Normalize
    for k, v in querytf.items():
        querytf[k] = v / math.sqrt(totalquerytf)

    # Start calculating summation of weights for tokens in query for all documents
    for doc in common_documents:
        total = 0  # summation of weights
        # is pure or not, True if document contains all tokens on query
        presentInAllDocuments = True
        for queryToken in unique(queryTokens):
            querytfweight = querytf[queryToken]  # query weight
            try:
                # document weight if exists
                documentfweight = top10List[queryToken][doc]
            except:
                presentInAllDocuments = False   # if not exists then turn pure value to False
                documentfweight = top10List[queryToken][min(
                    top10List[queryToken], key=top10List[queryToken].get)]  # document weight now is the upper ceiling of min weight value for that token
            total += querytfweight * documentfweight
        # save in dict with the sum as well as pure or not
        docSimilarity[filenames[doc]] = [total, presentInAllDocuments]
    if (not docSimilarity):  # if we end up with a empty dict then that means query not found in docs
        return('None', 0)
    k = 0  # key with max weight
    maxv = 0  # Value of that max weight
    ispure = False  # Was it acquired purely
    for k, v in docSimilarity.items():
        if(v[0] > maxv):
            maxk = k
            maxv = v[0]
            ispure = v[1]
    if(ispure is False):
        return('fetch more', 0)  # If max value was acquired falsly
    if(k is not 0):
        # If max value is infact true and was all tokens are present in this doc
        return(maxk, docSimilarity[maxk][0])


#################
#     MAIN      #
#################
# Open the directory and read each file one by one
for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    filenames.append(filename)
    doc = file.read()
    file.close()
    # Process data for one file
    processedData = preprocessing(doc, stopwordSorted)
    wordListAll.append(processedData)

# words is a dict of all words in all documents and their values are the index of documents in which they appear
for i in range(len(wordListAll)):
    docTokens = wordListAll[i]
    for token in docTokens:
        try:
            words[token].add(i)
        except:
            words[token] = {i}

# Change values of the word dict to be no of docs instead of list of docs {health: [doc1, doc2]} => {health: 2}
for i in words:
    words[i] = len(words[i])

# Calculate the tf-idf for all word tokens
for i in range(len(wordListAll)):
    totaltfidf = 0
    documenttokens = wordListAll[i]
    counter = Counter(documenttokens)
    for token in unique(documenttokens):
        tf = counter[token]
        tf = (1 + math.log(tf, 10)) if tf > 0 else 0
        df = document_frequency(token)
        idf = math.log(len(wordListAll)/df, 10)
        idfList[token] = idf
        tf_idf[i, token] = tf*idf
        totaltfidf += (tf_idf[i, token] * tf_idf[i, token])  # magnitude
    # Normalize
    for token in unique(documenttokens):
        tf_idf[i, token] = tf_idf[i, token] / math.sqrt(totaltfidf)

# Sort by highest weights first
tf_idf = OrderedDict(sorted(
    tf_idf.items(), key=operator.itemgetter(1), reverse=True))

############
# GRADING #
###########

graderTokenForIdf = "reason"
graderTokenForWeight = "hispan"
graderFileForWeight = "2012-10-16.txt"
graderQuery = "health insurance wall street"

print("%.12f" % getidf(graderTokenForIdf))
print("%.12f" % getweight(graderFileForWeight, graderTokenForWeight))
print("(%s, %.12f)" % query(graderQuery))
