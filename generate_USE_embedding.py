# generate_USE_embedding.py
# load text files and generate & save sentence embedding vectors for each row
# 4/2/20 hongmi lee

import glob
import tensorflow as tf
import tensorflow_hub as hub
import numpy 

# change directory names for your computer
rootdir = '/Users/hlee239/Desktop/remote2/scratch/hongmi/NarNet/text_analysis/'
textdir = 'textfiles/' % where text files are located
embeddir = 'USE_embeddings/'% directory where embedding csv files are saved 

alltxts = []
for file in glob.glob(rootdir + textdir + '*.txt'):
    file = file.split("/")
    file = file[len(file)-1]
    file = file[0:(len(file)-4)]
    alltxts.append(file)

# Import the Universal Sentence Encoder's TF Hub module
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)

for txtname in alltxts:
    
    # load and segment text file
    txtfile = open(rootdir + textdir + txtname + '.txt')

    alllines = []
    for line in txtfile:
        thisline = line.rstrip()
        if len(thisline) != 0:
            alllines.append(thisline)
    print(alllines)        

    # get embeddings
    with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      embeddings = session.run(embed(alllines))
      print(embeddings)    

    # save as csv
    fname = rootdir + embeddir + 'embeddings_' + txtname + '.csv'
    numpy.savetxt(fname,embeddings,delimiter=",")