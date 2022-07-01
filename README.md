# S2LD, Data Discovery using Natural Language Questions via a Self-Supervised Approach
S2LD is a self-supervised data discovery system that finds tables among a large collection given natural language questions. It consists of two separated stages:

1. offline, where tables in csv formats are converted to vectors and then indexed.  
2. online, where a user asks a question and it will return top 5 tables that most likely answers the question.

We use (changed) code from https://github.com/facebookresearch/FiD.git (OpenQA) and https://github.com/UKPLab/plms-graph2text.git (SQl2Question), big thanks to them.

To try our system, follow the steps in Setup

## 1. Setup
### 1.1. System requirements

Ubuntu 18.04(or above). GPU that supports CUDA 10.0 is needed to get good performance. CPU only also works but it will be very slow.

### 1.2. Prepare enviroment

a) Make sure the repository directory "open_table_discovery" in an empty parent directory
 
   A lot of data will be downloaded to the parent directory. If the parent directory has other subdirectories, there may be some collisions. After a new directory is created, move "open_table_discovery" to the new directory 
   
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

   There are two datasets, "fetaqa" which is smaller and "nq_tables" which is much larger. "fetaqa" is recommened to try.
     
   ```   bash
   ./get_data.sh fetaqa
   ```

## 2. Train
    
   ```   bash
   ./train.sh fetqa
   ```
    
## 3. Test
   ```   bash
   ./test.sh fetqa
   ```
    

