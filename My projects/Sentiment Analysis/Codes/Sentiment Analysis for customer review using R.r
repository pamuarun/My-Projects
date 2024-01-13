# load requried packages
library(tidyverse) 
library(syuzhet)
library(tidytext)

# import text dataset

df = read.csv("C:/Users/VAMSI/OneDrive/Desktop/R_project/flipkart.csv")
df
text.df = tibble(text = str_to_lower(df$Review))
text.df

 # analyze sentiments using the syuzhet package based on the NRC sentiment dictionary
emotions <- get_nrc_sentiment(text.df$text)
emo_bar <- colSums (emotions)
emo_sum <- data.frame(count=emo_bar, emotion=names(emo_bar))



# create a barplot showing the counts for each of eight different emotions and positive/negative rating
ggplot(emo_sum, aes (x = reorder (emotion, count), y = count)) +
  geom_bar(stat ='identity')



# sentiment analysis with the tidytext package using the "bing" lexicon
bing_word_counts = text.df %>% unnest_tokens (output = word, input=text) %>%
  inner_join(get_sentiments("bing")) %>% 
  count(word, sentiment,sort=TRUE)



#select top 10 words by sentiment

bing_top_10_words_by_sentiment = bing_word_counts %>%
  group_by(sentiment) %>%
  slice_max(order_by = n, n = 10) %>%
  ungroup() %>% 
  mutate(word = reorder (word, n))
bing_top_10_words_by_sentiment




# create a barplot showing contribution of words to sentiment
bing_top_10_words_by_sentiment %>%
  ggplot(aes (word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs (y = "Contribution to sentiment", x = NULL) +
  coord_flip()


#sentiment analysis with the tidytext package using the "loughran" lexicon

loughran_word_counts = text.df %>% unnest_tokens(output = word,input = text) %>%
  inner_join(get_sentiments("loughran")) %>%
  count(word, sentiment, sort = TRUE)



#select top 10 words by sentiment
loughran_top_10_words_by_sentiment = loughran_word_counts %>%
  group_by(sentiment) %>%
  slice_max(order_by = n, n = 10) %>%
  ungroup %>% 
  mutate (word = reorder (word, n))
loughran_top_10_words_by_sentiment










#create a barplot showing contribution of words to sentiment

loughran_top_10_words_by_sentiment %>% 
  ggplot (aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) + 
  facet_wrap (~sentiment, scales = "free_y") +
  labs (y = "Contribution to sentiment", x = NULL) +
  coord_flip()



# Bar plot
w <- rowSums(tdm)
w <- subset(w, w>=25)
barplot(w,
        las = 2,
        col = rainbow(50))



# Word cloud
library(wordcloud)
w <- sort(rowSums(tdm), decreasing = TRUE)
set.seed(222)
wordcloud(words = names(w),
          freq = w,
          max.words = 150,
          random.order = F,
          min.freq = 5,
          colors = brewer.pal(8, 'Dark2'),
          scale = c(5, 0.3),
          rot.per = 0.7)

library(wordcloud2)
w <- data.frame(names(w), w)
colnames(w) <- c('word', 'freq')
wordcloud2(w,
           size = 0.7,
           shape = 'triangle',
           rotateRatio = 0.5,
           minSize = 1)

# Sentiment analysis
library(syuzhet)
library(lubridate)
library(ggplot2)
library(scales)
library(reshape2)
library(dplyr)



# Read File

df = read.csv("C:/Users/VAMSI/OneDrive/Desktop/R_project/flipkart.csv")
df


tweets=iconv(df$Product_name)
tweets


# Obtain sentiment scores
s <- get_nrc_sentiment(tweets)
head(s)
tweets[4]
get_nrc_sentiment('delay')


# Bar plot
barplot(colSums(s),
        las = 2,
        col = rainbow(10),
        ylab = 'Count',
        main = 'Sentiment Scores for Filpkart Reviews')

accuracy <- conf_matrix$overall["Accuracy"]
cat("Accuracy: ", accuracy, "\n")


