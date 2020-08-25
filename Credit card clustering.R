library(factoextra)
library(dplyr)

cc_data <- read.csv("CC General.csv", row.names = 1)

str(cc_data)
summary(cc_data) ## All of our variables are numerical

#Checking for missing values
sapply(cc_data, function(x) sum(is.na(x))) ## 2 columsn have missing values

# As there is just 1 missing value in Credit_Limit, and only 0.3% of total missing in Minimum Payments, removing that row
cc_data <- cc_data[-which(is.na(cc_data$CREDIT_LIMIT)),]
cc_data <- cc_data[-which(is.na(cc_data$MINIMUM_PAYMENTS)),]

str(cc_data)
summary(cc_data)

## Scaling and applying PCA on the data
pr.ccdata <- prcomp(x= cc_data, scale = T , center = T)
summary(pr.ccdata)
biplot(pr.ccdata)

## Making a scree plot of PCA data
# Variability of each principal component: pr.var
pr.var <- pr.ccdata$sdev^2
# Variance explained by each principal component: pve
pve <- pr.var / sum(pr.var)

par(mfrow = c(1, 2))
# Plot variance explained for each principal component
plot(pve, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     ylim = c(0, 1), type = "b")

# Plot cumulative proportion of variance explained
plot(cumsum(pve), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     ylim = c(0, 1), type = "b")

# As 80% of variance in data is explained till PC7 , we'll take PC1-PC7 in our analysis

### Applying K-means clustering on the data
# Finding out an optimal value of k

wss <- 0

# For 1 to 20 cluster centers
for (i in 1:20) {
  km.out <- kmeans(pr.ccdata$x[, 1:7], centers = i, nstart = 20, iter.max = 50)
  # Save total within sum of squares to wss variable
  wss[i] <- km.out$tot.withinss
}

# Plot total within sum of squares vs. number of clusters
plot(1:20, wss, type = "b", 
     xlab = "Number of Clusters", 
     ylab = "Within groups sum of squares")

# Selectin k=8 based on the sum of squares plot, creating the final model
cc_data_model <- kmeans(pr.ccdata$x[, 1:7], centers = 8, nstart = 20, iter.max = 50)

table(cc_data_model$cluster)

plot(x= cc_data$BALANCE,y=cc_data$CREDIT_LIMIT,
     col = cc_data_model$cluster,
     main = paste("k-means clustering of Credit card data with 8 clusters"),
     ylab = "Balance", xlab = "Credit Limit")

## Visualize only the first 2 PCA variables based on clusters
fviz_cluster(cc_data_model, data = cc_data)

## Check the mean of all variables based on their clusters to get insight into customer segmentations formed
summary_clusters <- cc_data %>%
  mutate(Cluster = cc_data_model$cluster) %>%
  group_by(Cluster) %>%
  summarise_all("mean")


summary_clusters



