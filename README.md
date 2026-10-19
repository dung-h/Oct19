# Oct19

Research workspace for the Data Mining assignment on personalised paper recommendation.

The supplied competition data are in `assignment-data-mining-261.zip`; extract it at the repository root before running an experiment. `abstracts_collected.csv` is the consolidated metadata/abstract table used by the later experiments. The collection scripts at the repository root document how that table was assembled.

Each experiment is self-contained under `experiments/` and keeps its implementation, plan, and notes. Runtime artifacts such as model checkpoints, embeddings, API caches, and outputs are deliberately excluded from Git and should be regenerated on the remote GPU server.

For a fresh server setup, create an isolated Python environment, install the requirements file in the experiment you plan to run, then run that experiment's documented entry point. Several experiments require GPU-enabled PyTorch and Hugging Face model downloads.
