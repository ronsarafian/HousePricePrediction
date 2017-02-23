
# "House Price Prediction"
Ron Sarafian


The following is part of my submission for the [kaggle][1] House Prices competition. The goal is to
predict House price using advanced Machine Learning methods.

for more information see the competition site [here][2]

[1]:https://www.kaggle.com/
[2]:https://www.kaggle.com/c/house-prices-advanced-regression-techniques

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```


## Introduction

In this Analysis I use the [Ames Housing dataset][3] describing the sale of individual residential
property in Ames, Iowa from 2006 to 2010. The data set contains almost 3000 observations and a 
large number of explanatory variables involved in assessing home values. The challenge is to 
predict the final price of each house

The competition structure is as follow:

 - Use the *train.csv* file to train your model.
 - For each Id in the test.csv file predict the value of the *SalePrice* variable. 
 - Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the
 predicted value and the logarithm of the observed sales price
 
[3]:https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

-------


#### Packages

I use the following packages in this analysis:

``` {r packages, warning=F, message=F}
library(ggplot2)
library(caret)
library(dplyr)
library(corrplot)
library(RCurl)
library(lattice)
library(knitr)
library(plotly)
library(leaflet)
library(zoo)
library(reshape2)
```


### Understand the Data

In this section I try to get a better sense of the data, see if some cleaning and pre-processing 
methods are needed, and understand the patterns of NA values.

```{r workingDirectory, echo=F, results='hide'}
setwd("C:/Users/User/Desktop/fun/house")
```

Start by loading the raw data.

```{r readFiles}
train.raw <- read.csv("train.csv")
test.raw <- read.csv("test.csv")
```

The training data contain 1460 observations and 81 features. First let's have a look on the 
structure of the training dataset.

```{r dim, comment=""}
dim(train.raw)
str(train.raw)
```

The explanatories (columns 1:80) include 4 types of variables:

- Nominal
- Ordinal
- Discrete
- Continuous

As a first step, it would be more effective to classify the explanatories into these categories.
For this I use the dataset [codebook][4].

[4]:https://ww2.amstat.org/publications/jse/v19n3/decock/DataDocumentation.txt

```{r varsTypes, comment=""}
codebook <- getURL("https://ww2.amstat.org/publications/jse/v19n3/decock/DataDocumentation.txt")
# split the text file to strings:
codebook <- unlist(strsplit(codebook, "[ \n()]+"))
# utilizing the order of the variable description (identical to the variables order in the dataset).
# searching for the features' types pattern in the codebook text file:
grep.type <- grep("Nominal|Ordinal|Discrete|Continuous", codebook, value = T)
# arrange in a data.frame (ignoring the first elemet which relates to the observation index):
variables <- data.frame(name = names(train.raw), type = grep.type[2:82])
continuous <- as.character(variables[variables$type == "Continuous",1])
discretes <- as.character(variables[variables$type == "Discrete",1])
nominals <- as.character(variables[variables$type == "Nominal",1])
nominals <- nominals[2:24] # removing ID variable from the predictors
ordinals <- as.character(variables[variables$type == "Ordinal",1])
# first 10 variables' types
head(variables, 10)
```

Now, let's get a better sense of the data. It would be useful to evaluate variables from each 
type separately. First, lets see how many variables in each type group:

```{r tableVarTypes, comment=""}
table(variables$type)
```


#### Nominal predictors

Starting with the Nominal, let's look at the variables levels:

```{r plotNominal, comment="", results='hide', fig.width=9, fig.height=6}
par(mfrow = c(5,5), mar = c(2,1,3,3)) # adjusting  graphical parameters
sapply(names(train.raw[ , nominals]),
       function(x) barplot(table(train.raw[ ,x]), main = x))
```

It seems that some features have low variation across levels (e.g. *Street*, *LandContour*,
*Condition1*, and more. The concern here that these predictors may become zero-variance predictors
when the data are split into cross-validation/bootstrap sub-samples or that a few samples may have
an undue influence on the model. In the following section I will try to identify these near-zero
-variance predictors and adjust them before modeling. Also, the variable *MSSubClass* should be
converted into factor.

```{r}
train.raw$MSSubClass <- as.factor(train.raw$MSSubClass)
```

#### Ordinal predictors

In order to maximize the information that can be derived from the Ordinal variables it is 
important to express the order of the levels. I will do this in the following sections. For now
let's look at the variation across the levels

```{r plotOrdinal, comment="", results='hide', fig.width=9, fig.height=6}
par(mfrow = c(5,5), mar = c(2,1,3,3)) # adjusting  graphical parameters
sapply(names(train.raw[ , ordinals]),
       function(x) barplot(table(train.raw[ ,x]), main = x))
```

It can bee seen that in some variables the variation is relatively low. Again, near-zero-variance 
predictors need to be handled. It should be note that the variables level's originality is not 
necessarily represented in the barplots. 

#### Discrete predictors

few words...

```{r plotDiscrete, comment="", results='hide', fig.width=9, fig.height=4}
par(mfrow = c(3,5), mar = c(2,1,3,3)) # adjusting  graphical parameters
sapply(names(train.raw[ , discretes]),
       function(x) barplot(table(train.raw[ ,x]), main = x))
```

The discrete predictors histograms look fine.


#### Continuous predictors

In some continuous explanatories such as *PoolArea* or *BsmtFinSF1* a 0 value means that there is
no measurement (e.g. no pool or no finished basement), in this case it should be reported through
related categorical / Ordinal variable so that information does not get lost (I'll handle it 
later). For now, it will be more useful to look only on the distribution of positive values.

```{r plotContinuos, comment="", results='hide', fig.width=9, fig.height=5}
par(mfrow = c(4,5), mar = c(2,1,3,3)) # adjusting  graphical parameters
sapply(names(train.raw[ , continuous]),
       function(x) plot(density(train.raw[train.raw[ ,x]!=0,x],
                                na.rm = T), main = x))
```

It can be seen that some of the continuous variables distributions are very much skewed. I will
further address this issue in the Cleaning and PreProcessing section.


#### Missing values

Another important aspect in understanding the data is to understand the patterns of missing values
Here I provide a short analysis of missing values in the dataset.

I'll stars by calculating the number of NA values for each predictor:

```{r cormat}
vars.na.share <- sapply(train.raw, function(x) mean(is.na(x)))
variables$na.share <- vars.na.share
variables$na.exist <- vars.na.share > 0
par(mfrow = c(1,1))
na.table <- filter(variables, na.exist == T) %>% arrange(-na.share)
ggplot(data = na.table,
       aes(x = factor(name, levels = na.table$name),
           y = na.share)) +
    geom_bar(stat = "identity") + 
    coord_flip() +
    labs(title = "Share of NA values", x = "", y = "")
```

For many predictors the share of NA values is too significant to ignore, and it seems there is no
alternative than dig into the data.

Easy to see that for some variables the same share of NA values. Let's take care of them first.

Starting by *Garage* related variables, here are the first 6 NA rows:

```{r}
garage <- grep("Garage", names(train.raw), value = T)
kable(head(train.raw[is.na(train.raw$GarageType),garage]), 
      row.names = F, format = "markdown", align = "c")
```

We see that the *GarageCars* and the *GarageArea* are both 0 when NAs exist. By looking at the
codebook we can see that the common denominator is that NA represent **No Garage**
in all these variables.

Next, *Basemente* related variables:

```{r}
basement <- grep("Bsmt", names(train.raw), value = T)
kable(head(train.raw[is.na(train.raw$BsmtFinType1), basement]), 
      row.names = F, format = "markdown", align = "c")
```

Here also the codebook states that in all these variables NA values represent **No Basement**

Last group: *Masonry veneer* related variables:

```{r}
masvnr <- grep("MasVnr", names(train.raw), value = T)
kable(head(train.raw[is.na(train.raw$MasVnrType), masvnr]), 
      row.names = F, format = "markdown", align = "c")
```

These two are the only *Masonry veneer* related variable. There is no explanation in the codebook 
when the value is NA, and here *None* gets its own level. however the number of NA is negligible
(less than 0.5%). I'll assume that NA are "None" for *MasVnrType* and 0 for *MasVnrArea*.


The rest of the NAs and their description from the codebook by type:

-Ordinal / Nominal
 - FireplaceQu - Fireplace quality
 - Fence - Fence quality
 - Alley - Type of alley access to property
 - MiscFeature - Miscellaneous feature not covered in other categories
 - PoolQC - Pool quality

In all of these variables NA represent **None** (e.g. no fence, no pool, etc.)

- Continuous:
 - LotFrontage - Linear feet of street connected to property

Here we don't have an explanation of NA value but it is reasonable to assume that NA means 0 (e.g. no
street is connected). To check if it makes sense I look at the smallest values of *LotFrontage*: 

```{r smallLotFrontage}
quantile(train.raw$LotFrontage, 
         probs = seq(.01,.05,len = 5),
         na.rm = T)
```

There are reasonable small values that are not far from 0, so the assumption of NA = 0 make sense.

-------

### Data Cleaning and PreProcessing

In this section I am going to clean and arrange the data, and to prepare it for predictions. It is 
important to perform exactly the same actions of PreProcessing on the testing set for future 
predictions.

```{r newdf}
# starting by creating new dataframes for training and testings
train.clean <- train.raw
test.clean <- test.raw
```


#### Imputing Missing values

As I have shown, the pattern of missing values can be identify. Here I transform the missing data
into meaningful information, according to the description detailed in the *Missing values* section.

```{r missingValues}
## training set
    # factors
for (i in na.table[na.table$type %in% c("Ordinal","Nominal"),"name"]){
    train.clean[ ,i] <- `levels<-`(addNA(train.clean[ ,i]),
                          c(levels(train.clean[ ,i]), "None"))
}
    # integers
for (i in na.table[na.table$type %in% c("Continuous","Discrete"),"name"]){
    train.clean[ ,i][is.na(train.clean[ ,i])] <- 0
}

## testing set
    # factors
for (i in na.table[na.table$type %in% c("Ordinal","Nominal"),"name"]){
    test.clean[ ,i] <- `levels<-`(addNA(test.clean[ ,i]),
                          c(levels(test.clean[ ,i]), "None"))
}
    # integers
for (i in na.table[na.table$type %in% c("Continuous","Discrete"),"name"]){
    test.clean[ ,i][is.na(test.clean[ ,i])] <- 0
}

```


#### Skewness of continuos predictors

It can be seen that some of the continuous variables distributions (non-zero values) are very much
skewed. to cope whit this issue, taking a monotonic transformation of the the continues predictors
can be a great idea. It should be note that zero values represent no measurement, hence variables 
which contain zero values should be treated carefully. As mentioned, the absence of continuous 
measurement is already reported through other variable (e.g. PoolQC == "None" (means no pool), hence
PoolArea == 0). Thus, I would like to keep zeros when the categorical variable value is "None". To 
this I employ the The one-parameter **Box-Cox** transformation only for non-zeros values.

The one-parameter Box–Cox is a useful data transformation technique, used to stabilize variance, 
make the data more normal distribution-like, improve the validity of measures of association such as
the Pearson correlation between variables and for other data stabilization procedures. It defined
as:

$$ y_{i}^{(\lambda)} =
\begin{cases}
\frac{y_{i}^{\lambda} -1}{\lambda}, & \text{if }\lambda  \ne 0 \\
\ln{y_{i}}, & \text{if }\lambda =0
\end{cases} $$

where the power parameter $\lambda$ is estimated using the profile likelihood function. (if you are
not familiar with the Box-Cox transformation see this insightful [page][5])

[5]:http://onlinestatbook.com/2/transformations/box-cox.html

```{r boxcox, comment=""}
cont.pred <- continuous[-20] # transform predictors only
## training set
train.clean[ ,cont.pred][train.clean[ ,cont.pred]==0] <- NA # exluding zeros

for (i in names(train.clean[ ,cont.pred])) {
    boxcox <- BoxCoxTrans(train.clean[ ,i], na.rm = T)
    train.clean[ ,i] <- predict(boxcox, train.clean[ ,i])
train.clean[ ,cont.pred][is.na(train.clean[ ,cont.pred])] <- 0 # replacing zeros
}

## testing set
test.clean[ ,cont.pred][test.clean[ ,cont.pred][-81]==0] <- NA # exluding zeros

for (i in names(test.clean[ ,cont.pred[-81]])) {
    boxcox <- BoxCoxTrans(test.clean[ ,i], na.rm = T)
    test.clean[ ,i] <- predict(boxcox, test.clean[ ,i])
test.clean[ ,cont.pred][is.na(test.clean[ ,cont.pred])] <- 0 # replacing zeros
}
```


#### Near-Zero-Variation predictors

After data cleaning and before moving to prediction, excluding variables which have almost no
variation in a way they could be harmful to the prediction is a useful step. I employ the
*nearZeroVar* function to identify these predictors.

```{r}
nzvMat <- nearZeroVar(train.clean, saveMetrics = T)
nzvMat

nzvNames <- row.names(nzvMat[nzvMat$nzv==T, ])
train.clean <- train.clean[ ,!nzvMat$nzv]
test.clean <- test.clean[ ,!nzvMat$nzv[-81]]
```

-------


### Predictive analytics

As a first step, I divide the training data to 80% training, 20% testing/validation (note that
the the *ts* data is not part of the model). From this stage any learning will be using the
training set.

```{r partition}
set.seed(1948)
intrain <- createDataPartition(train.clean$SalePrice, p = 0.8, list = F)
tr <- train.clean[intrain, ]
ts <- train.clean[-intrain, ]
```


#### Pre-Modeling: Finding patters in the data

Before moving to the construction of the model, it is useful to find some patterns in the data to
get some hints about the sort of correlation between the variables. 

Let's start by plotting the correlation matrix of the continuous variables;

```{r corplot , fig.width=6, fig.height=6}
conttoplot <- continuous[!(continuous %in% nzvNames)]
corrplot(cor(tr[ ,conttoplot]), method = "ellipse", tl.col = "black")
```

As might be expected the correlation of variables related to house size have positive correlation 
with the price, and also to other size related variables.

Next, let's look at the relation between ordinal variables which relate to the quality 
 of elements in the house and the price

```{r quality, results='hide', fig.width=9, fig.height=6}
# define the order of quality levels:
q.order <- c("None", "Po", "Fa", "TA", "Gd", "Ex")
# find the relevant variables (Ex is one of the levels)
q.var <- sapply(tr, function(x) sum(levels(x)=="Ex")>0)
par(mfrow = c(3,3), mar = c(2,1,3,3))
for ( i in names(tr[ ,q.var])) {
    p <- tr[ , "SalePrice"]
    o <- factor(tr[ , i], levels = q.order)
    boxplot(p~o, main = i)
}
```

We can see the monotonic effect of levels, however some exceptions exist.

Let's look at the Overall quality and Conditions variables in a more fancy plot:

```{r overall, fig.width=9, fig.height=4}
OA.df <- tr[ ,c("OverallQual","OverallCond","SalePrice")]
pl.q <- plot_ly(OA.df, y = ~SalePrice, x = ~OverallQual, 
                type = "box", name = "Overall Quality")
pl.c <- plot_ly(OA.df, y = ~SalePrice, x = ~OverallCond, 
                type = "box", name = "Overall Condition")
subplot(pl.q, pl.c)
```

-------

It is interesting that while the Overall Quality correlation  with the Price is monotonic,
the Overall Condition correlation is less clear. We can see the large variation of House
Price in the some Condition levels (5 and 9). This may require to threat ordinal variables as 
categoricals o to use other techniques. 

Next, we should look for time trends and serial correlation that might need to be controlled. 

```{r, message=F, fig.width=9, fig.height=4}
time.df <- tr[ ,c("YrSold","MoSold", "SalePrice", "BedroomAbvGr")]
time.df$SalePrice <- time.df$SalePrice/100000
time.df$quarter <- cut(tr$MoSold,
                       breaks = c(0,3,6,9,12),
                       labels = c("Q1","Q2","Q3","Q4"))

time.df$YearQtr <- with(time.df, as.yearqtr(paste(YrSold,quarter)))

time.df$Bedrooms <- cut(time.df$BedroomAbvGr,
                        breaks = c(-1,2,3,Inf),
                        labels = c("0-2","3","4+"))

time.agg <- summarise(group_by(time.df, YearQtr, Bedrooms),
                      mean = mean(SalePrice),
                      sem = sd(SalePrice)/sqrt(n()),
                      sales = n())

ggplotly(ggplot(time.agg, aes(x = YearQtr, colour = Bedrooms))+
             geom_ribbon(aes(ymin = mean - sem,
                             ymax = mean + sem), fill = "grey70", alpha = 0.2) +
             geom_line(aes(y = mean))+
             facet_grid(Bedrooms~.) +
             labs(x = "", y = "") +
             coord_cartesian(ylim = c(1,3.5))) %>%
    layout(title = "Mean Sale Price ($100,000)")
```

It seems that the Seasonality effects are not very much clear.

Finally, Let's have a look on the spatial structure.
Thanks to [JulienSiems][6] that posted the GPS coordinates of the neighborhoods (based on the 
school location in the district) in his [script][7] we can look for spatial effect of house 
prices. For instance, we can use the distances between neighborhoods and key areas such as 
employment and shopping areas and other effects that can not be captured by neighborhoods dummies.

[6]:https://www.kaggle.com/juliensiems/discussion?sortBy=mostVotes&group=comments&page=1
[7]:https://www.kaggle.com/juliensiems/house-prices-advanced-regression-techniques/cleaning-new-features-gps-coordinates-included

```{r coordinates}
coordinates <- data.frame(Neighborhood = levels(tr[ ,"Neighborhood"]),
                          lat = c(42.062806, 42.009408, 42.052500, 42.033590, 42.025425,
                                  42.021051, 42.025949, 42.022800, 42.027885, 42.019208, 
                                  41.991866, 42.031307, 42.042966, 42.050307, 42.050207,
                                  42.060356, 42.051321, 42.028863, 42.033611, 42.035540, 
                                  42.052191, 42.060752, 42.017578, 41.998132, 42.040106),
                          
                          lng = c(-93.639963, -93.645543, -93.628821, -93.627552, -93.675741, 
                                  -93.685643, -93.620215, -93.663040, -93.615692, -93.623401,
                                  -93.602441, -93.626967, -93.613556, -93.656045, -93.625827, 
                                  -93.657107, -93.633798, -93.615497, -93.669348, -93.685131,
                                  -93.643479, -93.628955, -93.651283, -93.648335, -93.657032))
```

We can look at the geographical distribution of the house prices: Circle size represent the
number of sales in the Neighborhood and darker colors represent higher average price. 

```{r map, message=F}
nbh.price <- summarise(group_by(tr, Neighborhood),
                       sales = n(),
                       mean.price = mean(SalePrice))

coordinates <- merge(x = coordinates,
                     y = nbh.price, 
                     by = "Neighborhood",
                     all.x = T)

pal <- colorNumeric(palette = "Reds",
    domain = coordinates$nbh.price)

Ames <- leaflet(coordinates) %>% 
    addTiles() %>% 
    addCircles(lat = ~lat,
               lng = ~lng, weight = 10,
               radius = ~sales*8,
               color = ~pal(coordinates$mean.price)) %>%
    addMarkers(popup = ~paste(Neighborhood,", Mean:",
                              round(mean.price),sep = ""))
Ames
```


### Prediction 

In this section I'll present some machine learning technique to predict the price of a house.

The evaluation method of the models would be according to the competition rules:

$$RMSE = \sqrt{
\frac{2}{n}
\sum_{i=1}^{n} 
\ln{\frac{y_i}{\hat{y_i}}}
}$$

where $y_i$ is the observed house sale price and $\hat{y_i}$ is the predicted value.

#### Linear Model

Here I present only 3 embarrassingly simple linear models. Later, I will post more complex models 
involving complicated machine learning techniques

However, the best benchmark in many cases is the Linear Least Squares.

```{r , comment="", warning=F, message=F}
# predictiong log(price)
tr$logp <- log(tr$SalePrice)
ts$logp <- log(ts$SalePrice)

lm1 <- train(logp~., data = tr[,-c(1,60)], method = "lm")
ts$pr1 <- predict(lm1, newdata = ts)

qplot(y = ts$pr1, x =ts$logp) + 
    labs(x = "Observed log(Price)", y = "Predicted log(Price)") +
    geom_abline(intercept = 0, slope = 1)
```

We can see that:

- The Linear model prediction is not very far
- There is a concern to Heteroskedasticity
- Outliers may exist

The RMSE:

```{r}
postResample(ts$pr1, ts$logp)
```

Now, Let's use ordinal variables as discretes including polynomial effect (only variables which 
explicitly express quality or condition level)

```{r tonum, warning=F, message=F}
tonum <- function(df) {
    for (var in names(df[ ,q.var])) {
        df[ ,var] <- as.numeric(
            recode_factor(df[ ,var],
                          None="0",Po="1",Fa="2",TA="3",Gd="4",Ex="5"))
        df[ ,paste(var,"2",sep = "")] <- df[ ,var]^2
        df[ ,paste(var,"3",sep = "")] <- df[ ,var]^3
    }
    df
}

tr.2<-tonum(tr)
ts.2<-tonum(ts)


lm2 <- train(logp~.,
             data = tr.2[ ,-c(1,60)], method = "lm")
ts.2$pr2 <- predict(lm2, newdata = ts.2)
postResample(ts.2$pr2, ts.2$logp)
```

It seems that using the ordinal variables as categoricals (e.g. using dummy for each level) is
more useful in terms of RMSE, however גifferences are not significant.

For the third linear model I add some variables.

First I create the house and the garage *Age* at the time of sale instead of *YrSold*, *YearBuilt* and *GarageYrBlt*.Also, Let's create the time from last renovation, instead of *YearRemodAdd*.

```{r}
tr$houseage <- tr$YrSold - tr$YearBuilt
ts$houseage <- ts$YrSold - ts$YearBuilt

tr$houseage2 <- tr$houseage^2
ts$houseage2 <- ts$houseage^2

tr$garageage <- (tr$YrSold - tr$GarageYrBlt) * ifelse(tr$GarageYrBlt==0,0,1)
ts$garageage <- (ts$YrSold - ts$GarageYrBlt) * ifelse(ts$GarageYrBlt==0,0,1)

tr$timeremode <- tr$YrSold - tr$YearRemodAdd
ts$timeremode <- ts$YrSold - ts$YearRemodAdd
```

Now, let's have a look on the correlations of the log(SalePrice) and size variables to check
whether polynomial effects are needed.

```{r , results='hide', fig.width=9, fig.height=6}
size.var <- grep("SF|Area", names(tr), value = T)
par(mfrow = c(3,4), mar = c(3,3,2,2))
for (v in size.var) {
    d <- data.frame(var = tr[,v], p = log(tr$SalePrice))
    d <- filter(d, var != 0) %>% arrange(-var)
    plot(d, main = v)
    model <- lm(d$p ~ d$var + I(d$var^2))
    pred.line <- predict(model, d)
    lines(x = d$var, y = pred.line, col = "red", lwd = 2)
}
```

In order to avoid overfitting I use only linear effect.

also, we treat OverallCond as factors:

```{r , comment="", warning=F, message=F}
tr.3 <- tr
ts.3 <- ts

tr.3$MSSubClass <- as.factor(tr.3$MSSubClass)
ts.3$MSSubClass <- as.factor(ts.3$MSSubClass)

 #rename the data.frame
tr.3 <- select(tr.3, -c(Id,YrSold,YearBuilt,GarageYrBlt,YearRemodAdd,
                        SalePrice))
ts.3 <- select(ts.3, -c(Id,YrSold,YearBuilt,GarageYrBlt,YearRemodAdd))
```

The third linear model:

```{r, warning=F, message=F}
lm3 <- train(logp~., data = tr.3, method = "lm")
ts.3$pr3 <- predict(lm3, newdata = ts.3)
postResample(ts.3$pr3, ts.3$logp)
```

To be continued.

Ron Sarafian.

