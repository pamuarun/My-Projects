# Load required packages
library(tidyverse)
library(syuzhet)
library(caret)
library(e1071)  # for SVM
library(tm)

df = read.csv("C:/Users/VAMSI/OneDrive/Desktop/R_project/flipkart.csv")
df
text.df = tibble(text = str_to_lower(df$Review))
text.df


text.df$sentiment <- as.factor(ifelse(emotions$positive - emotions$negative >= 0, "positive", "negative"))
text.df$sentiment


# Data Preprocessing
clean_text <- function(text) {
  # Convert to lowercase
  text <- tolower(text)
  # Remove special characters, numbers, and punctuation
  text <- gsub("[^a-z\\s]", "", text)
  
  # Remove extra white spaces
  text <- gsub("\\s+", " ", text)
  
  return(text)
}


# Apply text cleaning
text.df$text <- sapply(text.df$text, clean_text)
text.df$text




# Split the dataset into training and testing sets
set.seed(123)
splitIndex <- createDataPartition(text.df$sentiment, p = 0.7, list = FALSE)
train_data <- text.df[splitIndex, ]
test_data <- text.df[-splitIndex, ]
train_data
test_data



# Feature extraction (you might need to modify this based on your specific dataset and requirements)
train_features <- train_data$text
test_features <- test_data$text
train_features
test_features



# Tokenization and creating a document-term matrix
tokenize <- function(text) {
  word_tokens <- unlist(str_split(text, "\\s+"))
  return(word_tokens)
}

train_tokens <- sapply(train_features, tokenize)
test_tokens <- sapply(test_features, tokenize)

# creating Document Matrix

corpus <- Corpus(VectorSource(train_tokens))
dtm <- DocumentTermMatrix(corpus)
dtm


# Train an SVM model
svm_model <- svm(as.matrix(dtm), as.factor(train_data$sentiment))


# Make predictions on the test set
test_dtm <- DocumentTermMatrix(Corpus(VectorSource(test_tokens)), control = list(dictionary = Terms(dtm)))
predictions <- predict(svm_model, newdata = as.matrix(test_dtm))
predictions


# Evaluate the model
conf_matrix <- confusionMatrix(predictions, as.factor(test_data$sentiment))
print(conf_matrix)


# Accuracy
accuracy <- conf_matrix$overall["Accuracy"]
cat("Accuracy: ", accuracy, "\n")


# ROC Curve
roc_curve <- roc(test_data$sentiment, as.numeric(predictions == "positive"))
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)

# AUC (Area Under the Curve)
auc_value <- auc(roc_curve)
cat("AUC: ", auc_value, "\n")



# Precision and Recall
precision <- conf_matrix$byClass["Pos Pred Value"]
recall <- conf_matrix$byClass["Sensitivity"]
cat("Precision: ", precision, "\n")
cat("Recall (Sensitivity): ", recall, "\n")

# F1 Score
f1_score <- 2 * (precision * recall) / (precision + recall)
cat("F1 Score: ", f1_score, "\n")

# Visualize the confusion matrix
conf_matrix_table <- as.table(conf_matrix)
conf_matrix_plot <- ggplot(data = as.data.frame(conf_matrix_table),
                           aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = n), color = "red") +
  theme_minimal() +
  labs(title = "Confusion Matrix",
       x = "Reference",
       y = "Prediction")
print(conf_matrix_plot)




# Histogram of Sentiment Distribution
sentiment_histogram <- ggplot(text.df, aes(x = sentiment, fill = sentiment)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Sentiment Distribution",
       x = "Sentiment",
       y = "Count")

print(sentiment_histogram)