# Solo, Data Discovery using Natural Language Questions via a Self-Supervised Approach
Solo is a self-supervised data discovery system that finds tables among a large collection given natural language questions. It automatically generates training dataset from the target table collection and then trains the relevance model.
The system consists of two separated stages:

Offline stage 

   First, Tables in csv formats are converted to vectors and indexed. Second, during training, SQLs are automatically sampled from the table collection and translated to questions. Then traning questions and tables are automatically collected to train the system.     

Online stage

   A user asks a question and it will return top 5 tables that most likely answers the question.

To try the system, follow the following steps,

## 1. Setup
### 1.1. System requirements

Ubuntu 18.04(or above). GPU that supports CUDA 11.0 (above) and pytorch 1.12.1 is needed.

If possible, use solid state drive (SSD). Disk storage should be more than 200 G if you want to try all data released. 

### 1.2. Prepare enviroment
a) Install Conda from https://docs.conda.io/en/latest/

b) Make sure the repository directory "open_table_discovery" in an empty work directory <work_dir>
 
   A lot of data and other code will be downloaded to <work_dir>. If <work_dir> has other child directories, there may be some collisions. After <work_dir> is created, move the "open_table_discovery" directory in <work_dir> 
   
c) Create a new session using *tmux* or *screen*
   ```   bash
   tmux new -s [session name] or screen -S [session name] 
   ```
   It takes time to run the following scripts, so it is better to create a session.


d) Go to the "open_table_discovery" directory and run the following script in the new session.
   ```   bash
   ./setup.sh
   conda activate s2ld
   ```
e) Download models
   ```   bash
   ./get_models.sh
   ```
f) Download data

   There are 3 datasets, "fetaqa", "nq_tables" and "chicago_open". 
   "fetaqa" is recommended to try.
   To download, run
   ```   bash
   ./get_data.sh <dataset>
   ```
   Each dataset is corresponding to a directory "<work_dir>/data/<dataset>". 

## 2. Index tables
   We provide indexed "fetaqa", "nq_tables" and "chicago_open" by the "get_data.sh" script. 
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
   You can also reindex these 3 datasets by running the script.

## 3. Train
   We provide pretrained models for "fetaqa" and "nq_talbes" and 
   you can ignore this if you just want to try table discovery (section 5).  
    
   The default batch size is 4, if the GPU memory is less than 24 G, use a smaller value (one by one) by updating "train_batch_size" in file "system.config". 
   
   By default, Incremental Training each time generates a dataset with 1,500 questions.
   
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
   We have pretrained models for "fetaqa" , "nq_tables" and "chicago_open" in 
   "<work_dir>/models/\<dataset\>" with file name "\<dataset\>_relevance.pt". 
   If you retrain "fetaqa" or "nq_tables", "<work_dir>/models/\<dataset\>" will have a best model for each training. 
   The script always loads the recent model (by create time), 
   so if you want to use the pretrained models, move the other model in some other directory.
    
## 5. Interactive Web application 
   The system provides a web interface.
   The user inputs a question and then top 5 tables are returned and displayed. 
   To try the application, follow the steps,

### 5.1. Start web server 
   run the script with the dataset "chicago_open"
   ```   bash
   ./run_server.sh chicago_open
   ```
   If the script is run a local machine where you can use a browser on it, 
   open "http://127.0.0.1:5000" and then go to section 5.3
   
### 5.2. Client setting 
   If the server is on a remote machine, do port routing on the client machine by runing the following script,
   ```   bash
   ssh -N -f -L 127.0.0.1:5000:127.0.0.1:5000 <user>@<remote server>
   ```
   Then on client, open "http://127.0.0.1:5000".
    
### 5.3. Try demo 
   Type a question to try, e.g.
   
   What are the business hours of Employment Resource Center at Howard?



