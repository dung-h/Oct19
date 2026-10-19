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

Submit Prediction
Assignment Data Mining - 261
Recommend what to read next prioritize relevant papers while providing broad coverage of the papers.


Overview
Your task is to build a method that recommends what researchers should read next by assigning a priority to every paper in the test collection.

A successful reading order should:

Bring relevant papers forward: papers with higher reading relevance should appear earlier.

Provide broad coverage: as researchers continue reading, the list should represent different parts of the collection rather than repeatedly presenting similar content.

You will receive 2,494 training papers with existing ratings from 1 to 5, where higher ratings indicate higher reading priority. You will also receive 596 test papers whose ratings are hidden. Available metadata includes the paper’s title, venue, publication year, authors, and DOI.

Submit a CSV containing the columns id,priority, with exactly one row for each test paper. Higher priority values mean earlier reading. You may use any finite numeric scale; evaluation uses the resulting order without converting your scores into predicted 1–5 labels.

The competition score combines 60% normalized relevance and 40% normalized coverage, with coverage measured through title similarity across all reading-list lengths. The Public and Private leaderboards each evaluate 298 hidden test papers independently. Final competition ranking uses the Private leaderboard.

Your solution must use at least one substantive approach taught in class: classification, regression, clustering, or itemset-based recommendation. Additional techniques are welcome when justified.

Teams may contain up to five members. Alongside your predictions, submit a report and GitHub repository explaining your data analysis, insights, solution design, evaluation, and how to reproduce your submission. Consult the Evaluation and Rules pages for the complete metric and grading requirements.

For more details, please read the evaluation section and Rules page.

Start

a day ago
Close

2 months to go
Description
Data
Training data: 2,494 papers with metadata and an existing Label from 1 to 5. Higher labels indicate higher reading priority.
Test data: 596 papers with metadata and hidden labels split into Private and Public Test set.
Paper metadata: id, title, venue, year, authors, and doi.
Use the available data to understand the collection, identify useful patterns, and design your prioritization system. No additional labeling is required.

Course requirements and grading
Teams may have up to five members. Your implemented solution must use at least one substantive approach taught in class: classification, regression, clustering, or itemset-based recommendation using frequent itemsets or association rules. You may combine approaches and use additional techniques beyond the course scope.

Submit your Kaggle predictions, report, and GitHub repository, with instructions for reproducing the selected submission. Your report should connect domain understanding and data analysis to insights, solution choices, implementation, fair evaluation, error analysis, and conclusions.

The final assignment grade is:

60% leaderboard ranking grade + 40% report grade.

The ranking grade is converted from your final Private leaderboard position using the bands on the Rules page; it is not the raw competition score multiplied by 10. These course grading weights are separate from the relevance and coverage weights above.

Follow the deadlines and submission limits announced on Kaggle and the LMS. Report grades and final assignment grades will be announced approximately two weeks after the assignment ends.

See the Rules page for the full requirements and report rubric.

Evaluation
What to submit
Submit a CSV containing exactly the columns id,priority, with one row for each of the 596 test papers:

id,priority
1488,9.25
2524,0
2344,7.12
997,8.50
These rows illustrate the format; your submission must include every test ID exactly once. Each priority must be a finite, nonnegative number:

Priority 0: do not recommend the paper.
Positive priority: recommend the paper; higher values mean earlier reading.
You may use any nonnegative numeric scale; [0, 10] is recommended. Submitted priorities determine selection and reading order. They are not converted into predicted ratings 1–5.

In this example, papers 1488, 997, and 2344 are recommended in that order if they belong to the same evaluation split. Paper 2524 is excluded.

Competition evaluation
Submissions are evaluated using the harmonic mean of relevance and coverage. Scores range from 0 to 1, and higher is better.

The hidden ground-truth ratings represent reading preferences: rating 1 means “do not read,” while ratings 2–5 indicate increasing reading priority.

Selection and reading order
The test papers are divided into a hidden Public set of 298 papers and a hidden Private set of 298 papers. Students do not need to know which papers belong to each set.

Within each set, the evaluator:

Excludes papers with priority 0.
Sorts the remaining papers by descending priority.
Resolves tied priorities by ascending paper ID, treated as text.
Assigns reading positions starting at 1 and calculates relevance and coverage.
There is no fixed shortlist cutoff: every positive-priority paper is included in the recommended list for its split.

Relevance
Relevance measures the usefulness of the recommended papers, with greater weight given to earlier positions.

For a recommended paper at position (r), let (y_r) be its hidden true rating. Define its usefulness gain and position weight as:

[ g_r=\frac{y_r-1}{4}, \qquad w_r=\frac{1}{\log_2(r+1)}. ]

The gains for ratings 1, 2, 3, 4, and 5 are respectively 0, 0.25, 0.50, 0.75, and 1.

For (m) recommended papers, relevance is:

[ Q=\frac{\sum_{r=1}^{m}w_rg_r} {\sum_{r=1}^{m}w_r}. ]

This means:

Placing a higher-rated paper before a lower-rated paper improves relevance.
A recommended rating-1 paper contributes zero gain but still contributes its position weight to the denominator.
Swapping papers with equal true ratings does not change relevance.
An empty recommendation list has relevance 0.
Relevance measures both usefulness and reading order; it is not normalized against an ideal ground-truth ranking. A perfectly ordered list containing ratings below 5 can therefore have relevance below 1.

Coverage
Coverage measures recall of worthwhile papers, where worthwhile means a true rating of 2–5:

[ R= \frac{\text{Number of recommended papers with true rating greater than 1}} {\text{Total number of papers with true rating greater than 1 in the evaluated split}}. ]

Worthwhile papers assigned priority 0 remain in the denominator. Recommending a rating-1 paper adds no coverage. Reordering the same selected papers does not change coverage.

If a split contains no worthwhile papers, coverage is defined as 0. Coverage uses paper IDs and hidden ratings; title similarity is not part of the metric.

Final score
The final score combines relevance (Q) and coverage (R) with equal importance using their harmonic mean:

[ \boxed{\text{Score}=\frac{2QR}{Q+R}}. ]

If either component is zero, the final score is defined as 0.

For example, relevance of 0.80 and coverage of 0.60 produce:

[ \text{Score} =\frac{2(0.80)(0.60)}{0.80+0.60} \approx 0.685714. ]

This encourages useful, well-prioritized recommendations while recovering a substantial proportion of worthwhile papers.

Public and Private leaderboards
The Public leaderboard provides feedback during the competition. Final competition ranking uses the Private leaderboard. Public and Private scores are calculated independently and are not averaged for the final ranking.

Course assignment grading, including the report grade, is calculated separately.

Competition Host
TU VU NGOC

Prizes & Awards
Kudos

Does not award Points or Medals

Participation
13 Entrants

2 Participants

1 Team

4 Submissions

Tags
Custom Metric
Table of Contents
Rules accepted. Good luck!