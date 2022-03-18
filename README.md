# American Name Classifier
Predicts whether a sentence is an american name or not based on statistics rules created from a database of random names.

# Requirements
- Python3.6+  
- names==0.3.0  
- pandas==1.4.1   

Install: `pip3 install -r requirements.txt`

# Usage
 - To compute perfomance metrics using `test_database.csv`: `python3 main.py`;
 - To check whether if arguments are names or not: `python3 main.py "Charlie Chaplin" "The Last Tree"`

## Perfomance Metrics Output

```
## Generating a database of random names... ##
0 of 20000
...
19500 of 20000
Done

## Evaluating test_database.csv... ##
0 of 466
...
450 of 466
Done

## Wrong cases ##
Names predicted as non name:
                sentence  name  probability  predicted
25            Kwame Ture     1          0.5        0.0
71           Deb Haaland     1          0.5        0.0
81    Arianna Huffington     1          0.5        0.0
97           Josh Hawley     1          0.5        0.0
118       Othniel Looker     1          0.5        0.0
119       Lorde Cornbury     1          0.5        0.0
122     Lincoln MacVeagh     1          0.5        0.0
148        Nima Kulkarni     1          0.5        0.0
216  Lemanu Peleti Mauga     1          0.5        0.0
232         Orson Welles     1          0.5        0.0
237        Buster Keaton     1          0.5        0.0
246          Greta Garbo     1          0.5        0.0

Non names predicted as name:
             sentence  name  probability  predicted
276   West Side Story     0         0.83        1.0
318      King Richard     0         1.00        1.0
348  Rose Plays Julie     0         0.83        1.0
363       Black Widow     0         0.75        1.0
378      Palm Springs     0         1.00        1.0
395     Saint Frances     0         0.75        1.0
408    Corpus Christi     0         0.75        1.0
425         City Hall     0         0.75        1.0

## Metrics ##
True Positives=255
True Negatives=191
False Positives=8
False Negatives=12

General Accuracy=0.96
Names Accuracy=0.96
Non Names Accuracy=0.96

Precision=0.97
Recall=0.96
F1=0.96
```
