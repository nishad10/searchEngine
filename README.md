# Search Engine
>Mimicking a search engine that takes a query and returns a closest matching document

This program takes a query and finds for you a document from a list of documents that matches with your query. This is done by calculating the tf-idf weights of all words/tokens and the calculating the cosine similarity score in between the document and query tokens. 

# Usage
This program accepts two arguments through commandline
Argument1 = Query
Argument2 = Path to directory containing all documents

```python
python3 search_engine.py 'terror attack' ./presidential_debates/
```
Outputs => (2004-09-30.txt, 0.026893338131)

```python
python search_engine.py 'vector entropy' ./presidential_debates/
```
Outputs => (None, 0.000000000000)

### Output Format
We get a tuple as output where the first value is the name of the document that we think matches best with the query followed by the cosine similarity socre of that document.

If the query words do not appear in any documents you get None and 0 as outputs.

# Setup

`git clone https://github.com/nishad10/searchEngine.git`

`cd searchEngine`

`pip install --user -U nltk`

If you have trouble installing nltk then take a look here https://www.nltk.org/install.html

You need nltk in order to run this program. Using python3 is recommended but as far as I have tried it seems to work fine with python2.

