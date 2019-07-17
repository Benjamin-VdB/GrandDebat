from nltk.corpus import stopwords
# Initialize spacy french model, keeping only tagger component (for efficiency)
# nlp = spacy.load('fr', disable=['parser', 'ner'])
import spacy
spacy_nlp = spacy.load('fr_core_news_md')
import re
import pandas as pd

# Wordcloud
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel



# Initial cleaning, Tokenisation and cleaning special characters and punctuation
def sent_to_words(sentences):
    # Remove Emails
    sentences = [re.sub('\S*@\S*\s?', '', sent) for sent in sentences]
    # Remove new line characters
    sentences = [re.sub('\s+', ' ', sent) for sent in sentences]
    # Remove distracting single quotes
    sentences = [re.sub("\'", " ", sent) for sent in sentences]
    sentences = [re.sub("\’", " ", sent) for sent in sentences]
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization

# removing french stop words from the NLTK list
def remove_stopwords(texts):
    stop_words = stopwords.words('french')
    stop_words.extend(['plus', 'faire', 'tout', 'oui', 'non'])
    #spacy_stopwords = spacy.lang.fr.stop_words.STOP_WORDS
    #stop_words.extend(spacy_stopwords)  
    #print(stop_words)
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, texts_nostops):
    bigram = gensim.models.Phrases(texts, min_count=3, threshold=50) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts_nostops]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = spacy_nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Visualize the topics with wordclouds
def cloud6topics(stop_words,model):
    
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=15,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = model.show_topics(formatted=False)

    fig, axes = plt.subplots(3, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    
# Score the documents
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)



# LDA loop over the questions
#     - Data cleaning / lemmatization / bigrams / stopwords
#     - Dump the lemma
#     - Build the dictionnary, the corpus and the lda model for the documents
    
def lda_loop(first, last, df, num_topics, ):
    for q in range(first, len(questions)):
        print(df.columns[questions[q]])
        question_text.append(df.columns[questions[q]])

        # Selecting the column, removing NAs
        data = df.iloc[:,questions[q]].dropna().values.tolist()
    #     # Remove Emails
    #     data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    #     # Remove new line characters
    #     data = [re.sub('\s+', ' ', sent) for sent in data]
    #     # Remove distracting single quotes
    #     data = [re.sub("\'", " ", sent) for sent in data]
    #     data = [re.sub("\’", " ", sent) for sent in data]

        # Initual cleaning, tokenisation and cleaning special characters and punctuation (gensim simple prepocess)
        data_words = list(proc_func.sent_to_words(data))

        # Do lemmatization with spacy keeping only noun, adj, vb, adv
        data_lemmatized = proc_func.lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_lemmatized, min_count=3, threshold=50) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_lemmatized], threshold=50)  

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # Form Bigrams
        data_words_bigrams = [bigram_mod[doc] for doc in data_lemmatized]

        # Remove Stop Words
        data_words_nostops = proc_func.remove_stopwords(data_lemmatized)

        data_lemma.append(data_words_nostops)

        # save data lemmatized
        pickle.dump(data_lemma[q-first], open('./lda_models/data_lemma_ecolo'+ str(q) +'.pkl', 'wb'))

        # Create Dictionary
        id2word = corpora.Dictionary(data_words_nostops)

        # Create Corpus
        texts = data_words_nostops

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # Fit the LDA model with gensim for a number of predifined topics
        model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics,
                                               passes=10,
                                               per_word_topics=True)
        model_list.append(model)
        # save the model
        model_list[q-first].save('./lda_models/model_ecolo'+ str(q) +'.gensim')

        # score each document and merge back the id
        df_topic_sents_keywords = proc_func.format_topics_sentences(ldamodel=model, corpus=corpus, texts=texts)
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

        doc_scored.append(pd.concat([df.iloc[:,[7,questions[q]]].dropna().reset_index(drop=True), df_dominant_topic], axis=1))
        pickle.dump(doc_scored[q-first], open('./lda_models/doc_scored_ecolo'+ str(q) +'.pkl', 'wb'))

        # Visualize the topics with LDAvis
    #     pyLDAvis.enable_notebook()
    #     vis = pyLDAvis.gensim.prepare(model, corpus, id2word)
    #     vis
    #     LDAvis.append(vis)
    #     pyLDAvis.save_html(LDAvis[q],'./lda_vis/model_ecolo'+ str(q) +'.html')

        # Visualize WordCloud
        proc_func.cloud6topics(stop_words,model)

