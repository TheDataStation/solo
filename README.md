# S2LD, Data Discovery using Natural Language Questions via a Self-Supervised Approach
This is the repository for the paper [Data Discovery using Natural Language Questions via a Self-Supervised Approach](https://arxiv.org/abs/2301.03560).
S2LD is a self-supervised data discovery system that finds tables among a large collection given natural language questions. It automatically generates training dataset from the target table collection and then trains the relevance model.
The system consists of two separated stages:

Offline stage 

   First, Tables in csv formats are converted to vectors and indexed. Second, during training, SQLs are automatically sampled from the table collection and translated to questions. Then traning questions and tables are automatically collected to train the system.     

Online stage

   A user asks a question and it will return top 5 tables that most likely answers the question.

To try the system, follow the following steps,

## 1. Setup
### 1.1. System requirements

Ubuntu 18.04(or above). GPU that supports CUDA 10.0 (above) is needed.

If possible, use solid state drive (SSD). Disk storage should be more than 200 G if you want to try all data released. 

### 1.2. Prepare enviroment

a) Make sure the repository directory "open_table_discovery" in an empty work directory <work_dir>
 
   A lot of data and other code will be downloaded to <work_dir>. If <work_dir> has other child directories, there may be some collisions. After <work_dir> is created, move the "open_table_discovery" directory in <work_dir> 
   
b) Create a new session using *tmux* or *screen*
   ```   bash
   tmux new -s [session name] or screen -S [session name] 
   ```
   It takes time to run the following scripts, so it is better to create a session.

c) Go to the "open_table_discovery" directory and run the following script in the new session.
   ```   bash
   ./setup.sh
   ```
d) Download models
   ```   bash
   ./get_models.sh
   ```
e) Download data

   There are two datasets, "fetaqa" which is smaller and "nq_tables" which is much larger. 
   "fetaqa" is recommended to try.
   To download, run
   ```   bash
   ./get_data.sh fetaqa
   ```
   Each dataset is corresponding to a directory "<work_dir>/data/<dataset>". 

## 2. Index tables
   We provide indexed "fetaqa" and "nq_tables" by the "get_data.sh" script. 
   You can ignore this if you don't want to try indexing.
   
   To index a new dataset (table collection), create 2 diretories,
    
   a) <work_dir>/data/\<dataset\>
   
   b) <work_dir>/data/\<dataset\>/tables_csv 
   
   We expect each table is in csv format with the filename denoting the title (caption) 
   and all csv files are in "<work_dir>/data/\<dataset\>/tables_csv" or 
   its offspring directories to allow same file names.
   Then run the script
   ```   bash
   ./index.sh <dataset>
   ```
   You can also reindex "fetaqa" and "nq_tables" by running the script.

## 3. Train
   We provide pretrained models for "fetaqa" and "nq_talbes" and 
   you can ignore this if you just want to try table discovery (section 5).  
    
   The default batch size is 4, if the GPU memory is less than 24 G, use a smaller value (one by one) by updating "train_batch_size" in file "system.config". 
   
   By default, Incremental Training each time generates a dataset with 1,200 questions.
   
   If you want to retrain on "fetaqa" or "nq_tables", download the data first by "./get_data.sh". 
   
   If you want to train on other dataset, index the table collection first (section 2). 

   To train the relevance model, run
   ```   bash
   ./train.sh <dataset>
   ```
   After training, the best model will be deployed automatically to "<work_dir>/models/\<dataset\>" 

## 4. Test
   This is used to evaluate the retrieval accuracy of "fetaqa" and "nq_tables"
   ```   bash
   ./test.sh <dataset>
   ```
   We have pretrained models for "fetaqa" and "nq_tables" in 
   "<work_dir>/models/\<dataset\>" with file name "\<dataset\>_relevance.pt". 
   If you retrain "fetaqa" or "nq_tables", "<work_dir>/models/\<dataset\>" will have a best model for each training. 
   The script always loads the recent model (by create time), 
   so if you want to use the pretrained models, move the other model in some other directory.
    
## 5. Interactive demo 
   We use jupyter notebook to show the demo application.
   The user inputs the dataset and also a question and then top 5 tables are returned and displayed. 
   3 example questions are list for fetaqa. To try the demo, follow the steps,

### 5.1. Start demo server 
   run the script
   ```   bash
   ./run_demo.sh <dataset>
   ```
   If the script is run a local machine where you can use a browser on it, 
   follow the instruction on the console and 
   copy/paste the URL (starting with  http://localhost ...) into your browser. Then go to section 5.3
   
### 5.2. Client setting 
   If demo server is on a remote machine, do port routing on the client machine by runing the following script,
   ```   bash
   ssh -N -f -L 127.0.0.1:5000:127.0.0.1:5000 <user>@<remote server>
   ```
   Then on client, copy/paste the URL shown on the server console into your browser.
    
### 5.3. Try demo 
   Type a question to try.
   
