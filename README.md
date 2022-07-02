# S2LD, Data Discovery using Natural Language Questions via a Self-Supervised Approach
S2LD is a self-supervised data discovery system that finds tables among a large collection given natural language questions. It automatically generates training dataset from the target table collection and then trains a model. It consists of two separated stages:

1. Offline stage 

   First, Tables in csv formats are converted to vectors and indexed. Second, during training, SQLs are automatically sampled from the table collection and translated to questions. Then traning questions and tables are collected to train the system.     

2. Online stage

   A user asks a question and it will return top 5 tables that most likely answers the question.

We use code (changed) from https://github.com/facebookresearch/FiD.git (OpenQA) and https://github.com/UKPLab/plms-graph2text.git (SQl2Question), big thanks to them.

To try our system, follow the following steps 

## 1. Setup
### 1.1. System requirements

Ubuntu 18.04(or above). GPU that supports CUDA 10.0 is needed.

If possble, use solid state drive (SSD). Disk storage should be more thn 200 G if you wan to try all data released. 

### 1.2. Prepare enviroment

a) Make sure the repository directory "open_table_discovery" in an empty work directory <work_dir>
 
   A lot of data and other code will be downloaded to <work_dir>. If <work_dir> has other child directories, there may be some collisions. After <work_dir> is created, move the "open_table_discovery" directory in <work_dir> 
   
b) Create a new session using *tmux* or *screen*
   ```   bash
   tmux new -s [session name] or screen -S [session name] 
   ```
   It takes time to run the following scripts, so it is better to create a session.

c) Go to the "open_table_discovery" directory and run the folowing script in the new session.
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
    Each dataset correspondes to a directory "<work_dir>/data/<dataset>", 
    where <dataset> is the placehold for dataset

## 2. Indexing tables
   We provide indexed "fetaqa" and "nq_tables" by the "get_data.sh" script. 
   You can ignore this if you don't want to try indexing.
   To index a new dataset (table collection), create diretories 
   "<work_dir>/data/<dataset>" and "<work_dir>/data/<dataset>/tables_csv". 
   We expect each table is in csv format with the filename denoting the title (caption) 
   and all csv files are in "<work_dir>/data/<dataset>/tables_csv" or 
   its offspring directories to allow same file names.
   Then run the script
   ```   bash
   ./index.sh <dataset>
   ```
   You can also reindex "fetaqa" and "nq_tables" by runing the script.

## 3. Train
   The default batch size is 4, if the GPU memory is less than 24 G, use a smaller value (one by one) by editing "train_batch_size" in file "trainer.config". Incremental training is disabled by default to reduce training cost. If you want to enable incremental training, update "train_step_n" to 5000 in "trainer.config". The default question size is 10000 by "train_start_n" in "trainer.config"
   To train the relevance model, run
   ```   bash
   ./train.sh <dataset>
   ```
   After training, The best model will be deployed automatically to "<work_dir>/models/<dataset>" 

## 4. Test
    This is used for "fetaqa" and "nq_tables"
   ```   bash
   ./test.sh <dataset>
   ```
   We have pretrained models for "fetaqa" and "nq_tables" in 
   "<work_dir>/models/<dataset>" with file name "<dataset>_relevance.pt". 
   If you retrain "fetaqa" or "nq_tables", "<work_dir>/models/<dataset>" will have multiple models for each training. The script always loads the recent model (by create time), so if you want to use the pretrained models, mode the other mdoels in some other directory.
    
## 5. Interactive demo 
    To be continued 

