apple <- read.csv("C:/Users/VAMSI/OneDrive/Desktop/R_project/apple.csv", header = T)
apple
str(apple)


# Build corpus
library(tm)
corpus = iconv(apple$text)
corpus = Corpus(VectorSource(corpus))
inspect(corpus[1:5])


# Clean text
corpus <- tm_map(corpus, tolower)
inspect(corpus[1:5])



corpus <- tm_map(corpus, removePunctuation)
inspect(corpus[1:5])



corpus <- tm_map(corpus, removeNumbers)
inspect(corpus[1:5])


cleanset <- tm_map(corpus, removeWords, stopwords('english'))
inspect(cleanset[1:5])



removeURL <- function(x) gsub('http[[:alnum:]]*', '', x)
cleanset <- tm_map(cleanset, content_transformer(removeURL))
inspect(cleanset[1:5])


cleanset <- tm_map(cleanset, stripWhitespace)
inspect(cleanset[1:5])


# Term Document Matrix

tdm=TermDocumentMatrix(cleanset)
tdm


tdm=as.matrix(tdm)
tdm[1:10,1:15]


cleanset <- tm_map(cleanset, removeWords, c('aapl', 'apple'))
cleanset <- tm_map(cleanset, stripWhitespace)
inspect(cleanset [1:5])
           
# Term document matrix
tdm <- TermDocumentMatrix(cleanset)
tdm
           
tdm <- as.matrix(tdm)
tdm[1:10, 1:20]



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
           shape = 'Triangle',
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

apple <- read.csv("C:/Users/VAMSI/OneDrive/Desktop/R_project/apple.csv", header = T)
apple
tweets=iconv(apple$text)
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
        main = 'Sentiment Scores for Apple Tweets')



# Histogram of sentiment scores
hist(colSums(s), col = rainbow(10), main = 'Histogram of Sentiment Scores', xlab = 'Sentiment Score')

# Confusion matrix for sentiment analysis
actual_sentiments <- as.factor(ifelse(colSums(s) > 0, "Positive", ifelse(colSums(s) < 0, "Negative", "Neutral")))
predicted_sentiments <- as.factor(ifelse(colSums(s) > 0, "Positive", ifelse(colSums(s) < 0, "Negative", "Neutral")))
conf_matrix <- table(Actual = actual_sentiments, Predicted = predicted_sentiments)
conf_matrix

# Precision, Recall, and Accuracy
precision <- conf_matrix['Positive', 'Positive'] / sum(conf_matrix['Positive', ])
recall <- conf_matrix['Positive', 'Positive'] / sum(conf_matrix[, 'Positive'])
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Display precision, recall, and accuracy
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("Accuracy:", accuracy, "\n")

