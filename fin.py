#=========================================
#import libraries
#=========================================
import pandas as pd
import streamlit as st 
import string 
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


#=========================================
#creat data
#=========================================
@st.cache_data
def load_data() :
    data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\archive (6)\Reviews.csv")
    return data.head(1000)
#=======================================================
# cleaning
#=======================================================
def clean_data(text):
    return "".join(i.lower() for i in text if i not in string.punctuation)

#=======================================================
# remove stopwords
#=======================================================
def del_stop(text):
    stop_word = stopwords.words('english')
    words = text.split()
    return " ".join([i for i in words if i not in stop_word])
#=========================================
#creat data
#=========================================
st.set_page_config(page_title="NLP Dashboard", layout="wide")
st.title("🧠 NLP text processing")
st.write("This website contains reviews of some products where we prepared the data, filtered it, and then applied NLP to it to make it ready for the computer to understand the text.")

st.image(r"C:\Users\HP\Downloads\Gemini_Generated_Image_gqpck5gqpck5gqpc.png")
data = load_data()

#=========================================
#creat taps
#=========================================
t1,t2,t3,t4 = st.tabs(["What is NLP","General info about data","NLP","Text scales"])

#=========================================
#tab 1
#What is NLP
#=========================================
with t1 :
    st.title("What is NLP (Natural Language Processing) ?")
    st.write("Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that gives computers the ability to understand, interpret, and generate human language, whether written or spoken.Think of it as the translator between messy, complex human communication and the structured, logical language of machines (code and math).")
    st.title("Why Do We Use It ?")
    st.write("""We use NLP to bridge the gap between humans and technology. Here are the primary reasons:

Processing Massive Data: Humans can’t read millions of emails or social media posts in seconds; NLP can. It helps organizations analyze huge amounts of text to find trends or insights.

Structuring the Unstructured: Most human communication (emails, videos, chats) is "unstructured." NLP organizes this data into a format that computers can actually use for decision-making.

Breaking Language Barriers: Tools like Google Translate use NLP to allow people from different cultures to communicate instantly.

Efficient Human-Machine Interaction: It allows us to talk to our devices (like Siri, Alexa, or ChatGPT) using natural speech instead of complex programming commands.""")
#=========================================
#tab 2
#General info about data
#=========================================
with t2 :
    st.write("Number of reviwes : ",data.shape[0])
    # delete duplicates rows
    data.drop_duplicates(inplace=True)
    # delete null rows
    data.dropna(inplace=True)
    #missing val
    st.write("Missing values",data["Text"].isna().sum())
    #print first 5 reviwes
    st.header("First five reviwes")
    st.write(data.head())

#=========================================
#tab 3
#NLP
#=========================================
with t3 :

    #=======================================================
    # MAIN DATA
    #=======================================================
    st.title("🧠 NLP Text Processing Dashboard")
    st.subheader("📄 Original text")
    st.write(data["Text"].head(20))
    # processing
    data["clean data"] = data["Text"].apply(clean_data)
    data["Processed data"] = data["clean data"].apply(del_stop)

    st.subheader("🧹 Processed Text")
    st.write(data[["Text","clean data","Processed data"]].head(20))
#=========================================
#tab 3
#NLP
#=========================================
with t4 :
    #=======================================================
    # TF-IDF
    #=======================================================
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data["Processed data"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    st.subheader("🔢 TF-IDF table")
    st.dataframe(tfidf_df)
    
    
    
    top_words = tfidf_df.sum().sort_values(ascending=False).head(10)
    st.subheader("🔥 Top Important Words")
    st.bar_chart(top_words)









