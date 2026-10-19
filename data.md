Skip to
content


Create

Home

Competitions

Benchmarks

Game Arena

Data Hub

More

Your Work

Viewed


Assignment Data Mining - 261


faster_rcnn/inception_resnet_v2


The Pokémon Company - PTCG AI Battle Challenge Simulation


The Pokémon Company - PTCG AI Battle Challenge Strategy


AI Agent Security - Multi-Step Tool Attacks

Edited


notebookc8edb50d21


Continue Training phi3 phase 2


qwen7b


llm4waf_notebook


notebook7377ee7633


View Active Events

Search

Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
TU VU NGOC · Community Prediction Competition · Private · 2 months to go
Assignment Data Mining - 261
Recommend what to read next prioritize relevant papers while providing broad coverage of the papers.


Assignment Data Mining - 261

Submit Prediction
Dataset Description
Dataset Description
This dataset contains 3,090 research papers for a personalized recommendation task. Your goal is to identify papers worth reading and prioritize stronger recommendations in the reading order.

Each paper includes bibliographic metadata. The training data also provides a rating reflecting one researcher’s reading preferences.

Files
train.csv — 2,494 papers with metadata and reading-preference ratings.
test.csv — 596 candidate papers with metadata. Their ratings are hidden.
sample_submission.csv — the required submission template containing all 596 test IDs.
Columns
Column	Description
id	Unique paper identifier.
title	Paper title.
venue	Publication venue.
year	Publication year.
authors	Paper authors.
doi	Digital Object Identifier.
Label	Reading-preference rating from 1 to 5; available only in train.csv.
Reading-Preference Ratings
Rating 1 means the researcher does not want to read the paper. Ratings 2–5 indicate increasing reading priority, with 5 representing the strongest preference.

These ratings reflect personal reading interests, rather than universal judgments of scientific quality.

Submission Format
Submit a CSV with exactly two columns: id,priority. Include every test ID exactly once.

Priority 0: do not recommend the paper.
Positive priority: recommend the paper; higher values indicate earlier reading.
Priorities must be finite, nonnegative numbers. They do not need to use the 1–5 rating scale.

id,priority
1488,92.5
2524,0
2344,71.2
997,85.0
Evaluation
The score combines relevance and coverage using their harmonic mean:

Relevance rewards useful papers appearing earlier and penalizes recommendations with a true rating of 1.
Coverage measures recall: the proportion of all worthwhile papers—those rated 2–5—that are recommended.
The test set is divided into 298 Public and 298 Private papers. Membership is hidden, and recommendations are sorted and evaluated independently within each split. Final ranking uses the Private leaderboard.

See the Evaluation tab for the complete scoring formula.

Files
3 files

Size
549.25 kB

Type
csv

License
CC BY-NC-SA 4.0

Submission.csv(4.7 kB)

2 of 2 columns


id

priority
Label	Count
713.00 - 950.20	59
950.20 - 1187.40	74
1187.40 - 1424.60	55
1424.60 - 1661.80	57
1661.80 - 1899.00	60
1899.00 - 2136.20	63
2136.20 - 2373.40	54
2373.40 - 2610.60	65
2610.60 - 2847.80	53
2847.80 - 3085.00	56
713
3085
Label	Count
0.00 - 0.00	596
0
0
979
0
1106
0
1894
0
1718
0
1989
0
2010
0
2356
0
1179
0
3001
0
796
0
1771
0
986
0
1725
0
1296
0
2983
0
2666
0
2573
0
1999
0
1828
0
1877
0
2087
0
1581
0
1185
0
1873
0
952
0
2572
0
1835
0
1320
0
2083
0
1310
0
1053
0
1698
0
1372
0
2847
0
2537
0
2808
0
1311
0
2863
0
3085
0
1265
0
2446
0
2186
0
2898
0
2529
0
2810
0
997
0
2836
0
2194
0
1964
0
910
0
Data Explorer
549.25 kB

Submission.csv

Test.csv

train.csv

Summary
3 files

10 columns


Download All
Metadata
License
CC BY-NC-SA 4.0
Rules accepted. Good luck!