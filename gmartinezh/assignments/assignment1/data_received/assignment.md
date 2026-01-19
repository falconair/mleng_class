# Headlines to sentiment

One of our data scientists has built a sentiment model and your job is to deploy it. 

### What has the data scientist done?
The following zip file contains the contents of the project.
The various notebooks do the following:
1. Scrape headlines from NYT and Chicago Tribune
2. Convert headlines to vectors (using the SentenceTransformer library)
3. Build a scikit-learn SVM classifier to rate headlines as either Optimistic, Pessimistic or Neutral. The training is done by reading in a set of vectorized headlines and their corresponding labels. Once trained, the model is written to disk.
4. One notebook shows how to load this model and use it to score new headlines. <= This file is most relevant for your work.

### What is your task?
The model has already been trained and someone else will scrape the headlines. Your task is to:
1. Load headlines from a file
2. Convert them to vectors
3. Feed those vectors to the loaded SVM model and get outputs: Optimistic, Pessimistic, Neutral
4. Write the results in the following format: output label  COMMA  the original headline

### How will this be used?
On a daily basis, someone (or an automated program) will scrape headlines and run your program on them. They will then take the output of your program, aggregate it to a daily score and sell this "daily sentiment" to hedge funds, governments, etc.

### Program inputs and outputs
1. Create a program, called `score_headlines.py`
2. This program should accept an input text file which will contain one headline per line
3. This program should also accept a parameter describing the source of the headlines (example: chicagotribune, nyt)
4. If the client does not provide these two parameters, the program should give an appropriate and friendly error message
5. This program should output a new file, called `headline_scores_source_year_month_day.txt` (example: headline_scores_nyt_2025_01_15.txt)

example:

`python score_headlines.py todaysheadlines.txt nyt`

`python score_headlines.py headlines_from_la.txt la_times`

### Use git to sync your code between your laptop and the linux server
You should develop your code on your laptop, commit and push to GitHub, and pull from the linux server to make sure you can always track what version is on the linux server.

However, you may eventually want to transfer some files (perhaps test data) to the server, which do not belong in git. You can use the `scp` command ("secure copy") to copy files between your laptop and the linux server:

`scp local_file_name.txt yourusername@linuxserver:/remote_folder/remote_file_name.txt` <= Run this command on your laptop

This command is available on windows and mac terminals.

### Code review
Please make it a habit to run `pylint` (and, optionally, `black`) on your programs before committing changes to git. You don't have to follow every suggestion but you should be able to defend why you didn't. 

### Code submission for grade
You do not need to submit anything to canvas. I will check your code by actually running it on the linux server and inspecting the output. I will also review how you organized your code. 