                      #   Handwritten Digits Project

# Bring in the Data
trainData <- read.table(paste("http://archive.ics.uci.edu/ml/machine-learning-databases/",
                         "optdigits/optdigits.tra", sep=""), sep=",", header = FALSE, 
                          na.strings = c("NA", "", " "), 
                          col.names = c(paste("x", 1:64, sep=""), "digit"))

testData <- read.table(paste("http://archive.ics.uci.edu/ml/machine-learning-databases/",
                     "optdigits/optdigits.tes", sep=""), sep=",", header = FALSE,
                      na.strings = c("NA", "", " "),
                      col.names = c(paste("x", 1:64, sep=""), "digit"))

# check the dimension of the data
dim(trainData); dim(testData)

# Concatenate both data sets into one
digitDat <- rbind(trainData, testData); 

dim(digitDat)


###############################################
# Exploratory Data Analysis (EDA)
###############################################

digitDat0 <- data.matrix(digitDat[order(digitDat$digit),])

# Labels for plotting 
labs <- digitDat0[, c(65)] 

# Remove the known digits
digitDat0 <- digitDat0[,-c(65)] 

theUnique <- apply(digitDat0, 2, unique)
for (m in 1:length(theUnique)){
  if (length(unique(digitDat0[,m])) == 1)
    print(paste("x", m))
}

# delete x1 and x40 from the data
digitDat0 <- digitDat0[, -c(1, 40)]

n <- NROW(digitDat0)
color <- rainbow(n, alpha = 0.8)
heatmap(digitDat0, col=color, scale="column", Rowv=NA, Colv=NA,
        labRow=FALSE, margins=c(4,4), xlab="Image Variables", ylab="Samples",
        main="Heatmap of Handwritten Digit Data")


###############################################
# Principal Component Analysis (PCA)
###############################################

# Standardize the data
digitDat0.scaled <- scale(digitDat0, center=TRUE, scale=TRUE)

digitDat0.pca <- prcomp(digitDat0.scaled, retx=TRUE)

sd.pca <- digitDat0.pca$sdev
var.pca <- sd.pca**2
prop.pca <- var.pca/sum(var.pca)

plot(cumsum(prop.pca), type="h", xlab="Principal Component (PC)",
     ylab="Cumulative Proportion (CP)", main="Plot of PC vs CP")
abline(h=0.7, col='red')
abline(h=0.9, col='blue')


# output of the estimated firrst two PC directions 
a1.a2 <- digitDat0.pca$rotation[,1:2]
a1.a2


# Plot of PC2 versus PC1
library(RColorBrewer)
palette(brewer.pal(n=length(unique(labs)), name="Set3"))
plot(digitDat0.pca$x[,1:2], pch="", main="PC1 and PC2 for Handwritten Digits")
text(digitDat0.pca$x[,1:2], labels=labs, col=labs)
abline(h=0, v=0, lty=2)


library(MASS)
digitDat0.dist <- dist(digitDat0)
digitDat0.sam <- sammon(digitDat0.dist)

plot(digitDat0.sam$points[,1], digitDat0.sam$points[,2], t='n',
     main="Sammon's Nonlinear Mapping", xlab="First coordinate", 
     ylab="Second coordinate")
text(digitDat0.sam$points[,1], digitDat0.sam$points[,2], labels=labs, col=labs)
abline(h=0, v=0, lty=2)


library(Rtsne)
set.seed(121343)
tsne <- Rtsne(digitDat0, dims=2, perplexity=30, max_iter=500)
plot(tsne$Y, t='n', main="t-Distributed Stochastic Neighbour Embedding", 
     xlab="First tSNE coordinate", ylab="Second tSNE coordinate")
text(tsne$Y, labels=labs, col=labs)
abline(h=0, v=0, lty=2)