print("Pulse check data refresh initiated")
import docx
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#For word cloud
from wordcloud import WordCloud
from collections import Counter
from collections import Counter
from operator import itemgetter

#Import NLP libraries for sentiment analysis
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import TextBlob

# For Classfication
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def createtable(address):
    '''
    File name should be in a specific format as 
    '16th_PulseCheckMeeting_Group6_2023-03-15.docx'
    16th - number of plus check meeting
    PulseCheckMeeting - Name of the meeting
    Group6 - Group Number
    Date of meeting
    '''
    #Read the documents
    doc = docx.Document(address)

    #Read the document Name
    file_name = os.path.basename(address)
    values = file_name.split('_')
    
    #Create Data Dictionary
    data = {'Meeting_ID':[values[0]], 'Meeting':[values[1]],'Group':[values[2]], 'Date' :[values[3][:-5]]}

    #Create a New dataframe
    #df = pd.DataFrame(data, index=False)
    df = pd.DataFrame(data)
    
    #Convert the text data into text paragraphs
    text_para = []
    for para in doc.paragraphs:
        text_para.append(para.text)
    
    #From Text Data Extract Time stamp, name of speaker and the text
    time_stamp = []
    Speaker_Name = []
    Speaker_text = []
    for i in text_para:
        val = i.split('\n')
        time_stamp.append(val[0])
        Speaker_Name.append(val[1])
        Speaker_text.append(val[2])
    
    data_dict = {'time_stamp' : time_stamp, 'Speaker_Name': Speaker_Name, 'Speaker_text' : Speaker_text}

    #Create a New dataframe based on text values extracted above
    New_df = pd.DataFrame(data_dict)

    #concat with the dataframe and forward fill the values
    df = pd.concat([df, New_df], axis = 1)
    df.fillna(method = 'ffill', inplace = True)

    #Create a New column as set
    df['set'] = df.groupby('Speaker_Name').cumcount()+1
    
    #From Timestamp find the time in seconds speaker speek
    time_min = []
    time_max = []
    for i in range(len(df['time_stamp'])):
        A = df['time_stamp'][i]
        V = A.split(' --> ')
        t_min = V[0].split(':')
        t_max = V[1].split(':')
        time_min.append(float(t_min[0])*3600 + float(t_min[1])*60 + float(t_min[2]))
        time_max.append(float(t_max[0])*3600 + float(t_max[1])*60 + float(t_max[2]))
    
    time_diff_seconds = [time_max[i] - time_min[i] for i in range(len(time_min))]
    #Create a new column as time in seconds
    df['time_seconds'] = time_diff_seconds
    
    return df

def sentimenttextlength(df):
    #Merge the text Data based on Meeting ID and Text column
    sentiment_df = df.groupby(['Meeting_ID','Group','Speaker_Name'])['Speaker_text'].agg(' '.join).reset_index()
    # Find the lengh of text based on each speaker
    # Some speaker never spoke during the call, so to remove null we are adding hello as dummy text for all non speaker
    sentiment_df['Speaker_text'] = [s if s.lower().startswith("hello") else 'hello' +' '+ s for s in list(sentiment_df['Speaker_text'])] 
    # calculate the length of each row
    row_lengths = sentiment_df['Speaker_text'].str.len()
    # calculate the total length of all rows
    total_length = row_lengths.sum()
    # calculate the percentage of text in each row
    percentages = row_lengths/total_length * 100
    # add the column into the dataframe
    sentiment_df['text_percentage'] = percentages
    return sentiment_df

def time_spent(df):
    #Sum the time (in seconds) based on meeting ID and Speaker_Name
    time_df = df.groupby(['Meeting_ID','Group','Speaker_Name'])['time_seconds'].agg('sum').reset_index()
    final_df = pd.DataFrame()
    for i in list(time_df['Meeting_ID'].unique()):
        temp_df = time_df[time_df['Meeting_ID'] == i].reset_index(drop = True)
        for j in list(temp_df['Group'].unique()):
            temp_df1 = temp_df[temp_df['Group'] == j].reset_index(drop = True)
            total = temp_df1['time_seconds'].sum()
            temp_df1['time_percentage'] = [(temp_df1['time_seconds'][i]/total)*100 for i in range(len(temp_df1))]
            final_df = final_df.append(temp_df1,ignore_index = True)
    return final_df

def key_points(df, level):
    '''1 = Speaker wise, 2 = Meeting wise'''
    stop_words = ['think', 'on', 'not', 'they', "it's", 'Oh.', 'Ohh.','people','also', 'from', 'apart', 'There,', 'do', 'your','would', 'i', 'any', 'right?', 'then', 'but', 'yes', 'what', 'one','point','guys', "i'm", 'all', 'if', 'in', 'thing', 'was', 'or', 'because','get','now', 'just', 'no', 'my', 'should', 'me', 'could', 'at', 'only','go','giving', 'anything', "that's", 'very', "don't", 'is', 'can', 'ok',        'know', 'yeah', 'yeah.', 'it', 'are', 'be', 'for', 'this', 'with','kind',        'uh.', 'will', 'yeah,','give','make',"we'll",'again','coming','say','hey',        'are', 'of', 'you', 'that', 'we', 'to', 'have', 'a', 'hello', 'Uh','thank',        'Uh,', 'sure.', 'sure', 'So', 'the', 'some', 'some,', 'The.', 'The','see',        'there', "there's", 'the', 'Hi', 'hi', 'umm', 'guess', 'Guess', 'oh','got',        'um', 'uh', 'er', 'ah', 'like', 'well', 'and', 'so', 'right','which','something',        'literally', 'okay', 'totally', 'basically', 'actually','them', 'as', 'how',        'out', 'were','maybe','much','want','other','who','these','more','where','our',        'take','done','next','hmm','mmm','things','us','come'] + stopwords.words('english')
    if(level == 1):
        final_df = pd.DataFrame()
        for i in list(df['Speaker_Name'].unique()):
            temp_df = df[df['Speaker_Name'] == i].reset_index(drop = True)
            word_list = []
            for j in range(0, len(temp_df)):
                words = list(temp_df['Speaker_text'][j].split(" "))
                words = list(set(words).difference(stop_words))
                top_words = list((dict(sorted(Counter(words).items(), key = itemgetter(1), reverse = True)[:5])).keys())
                word_list.append(top_words)
            temp_df['Key_Words'] = word_list
            final_df = final_df.append(temp_df,ignore_index = True)
        return final_df
    elif(level == 2):
        final_df = pd.DataFrame()
        for k in list(df['Meeting_ID'].unique()):
            for l in list(df['Group'].unique()):
                temp_df = df[(df['Meeting_ID'] == k) & (df['Group'] == l)].reset_index(drop = True)
                total_text = temp_df['Speaker_text'].str.strip().str.replace(",", "").str.replace(".", "").tolist()
                try:
                    base = total_text[0]
                    for i in range(0, len(total_text)):
                        base = base + " " + total_text[i]
                    words = list(base.lower().split(" "))
                    filtered_list = []
                    for element in words:
                        if element not in stop_words:
                            filtered_list.append(element)
                    x = Counter(filtered_list)
                    top_words = pd.DataFrame((dict(sorted(x.items(), key=itemgetter(1), reverse=True)[:10])),index=[0]).T.reset_index().rename(columns={'index': 'Words',0: 'Occurences'})
                    top_words['Meeting_ID'] = k
                    top_words['Group'] = l
                    final_df = final_df.append(top_words,ignore_index = True)
                except:
                    pass
            final_df = final_df.append(top_words,ignore_index = True)
            final_df.insert(0,'Meeting_ID',final_df.pop('Meeting_ID'))
        return final_df

def sentiment_score_assign(df):
    sentiment_score = []
    for sentence in list(df['Speaker_text']):
        fv = [",",".","!","?","-","_","@","#"]
        for val in fv:
            sentence = sentence.replace(val,"")
        blob = TextBlob(sentence)
        for sentence in blob.sentences:
            score = sentence.sentiment.polarity
            sentiment_score.append(score)
    df['Sentiment_score'] = sentiment_score
    sentiment = []
    for i in range(len(df['Sentiment_score'])):
        val = df['Sentiment_score'][i]
        if val > 0:
            sentiment.append('positive')
        elif val < 0:
            sentiment.append('negative')
        else:
            sentiment.append('neutral')
    df['Sentiment'] = sentiment
    return df

def final_df_generation(folder_path):
    file_list = os.listdir(folder_path)
    data = []
    for file in file_list:
        df1 = createtable(folder_path + file)
        data.append(df1)
    # Concatenating all dataframe to a single dataframe
    df = pd.concat(data)
    # Sentiment Dataframe generation
    temp_df = sentimenttextlength(df)
    sentiment_df = sentiment_score_assign(temp_df)
    time_df = time_spent(df)
    print("Data ingestion started")
    topics_df = pd.read_excel("Input files\PC_topics.xlsx")
    print("Data ingestion completed")
    sentiment_df = pd.merge(sentiment_df, time_df, on = ['Meeting_ID','Group','Speaker_Name'],  how = 'inner')
    sentiment_df = pd.merge(sentiment_df, topics_df, on = 'Meeting_ID', how = 'left')
    sentiment_df.insert(1,'Topics_discussed',sentiment_df.pop('Topics Discussed'))
    sentiment_df = key_points(sentiment_df,1)
    key_df = key_points(sentiment_df,2)
    return sentiment_df, key_df

sentiment_df, key_df = final_df_generation('Input files/Transcript Documents/')

print("Storing refreshed data to directories")
# Storing files to respective directories
sentiment_df.to_excel('Output files/Pulse_check_data.xlsx')
key_df.to_excel("Output files/Keywords.xlsx")
print("Data refresh completed")
