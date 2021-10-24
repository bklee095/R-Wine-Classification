![Wine_Cultivated_Varietal_Classification_](https://user-images.githubusercontent.com/74638365/138602134-e0ca7239-7c72-4541-b9c2-f29fe7343233.png)

![GitHub last commit](https://img.shields.io/github/last-commit/bklee095/R-Wine-Classification)
# Italian Wine Cultivated Varietal Classification with Decision Tree


## Dataset

![Data source URL](https://archive.ics.uci.edu/ml/datasets/Wine)

This dataset is a compile of chemical and visual attributes of wines grown in the same region in Italy, but different types of cultivated varietals (cultivars). Cultivar is a subcategory below Species in biology. Cultivar is known to be a critical feature in wine production as it can affect everything about the product such as flavor profile, aroma, and aging potential. Some of the popular cultivars are Cabernet Sauvignon, Pinot Noir, Shiraz, and Riesling.

The dataset has 13 different wine attributes with 178 observations. The response variable is a categorical variable that indicates the wine to be one of the three cultivars. 

## Data Cleansing

![1](https://user-images.githubusercontent.com/74638365/138602848-f767c67b-195b-4fd3-95c1-8fac628d5579.PNG)

The raw data lacks a header for indication of the attributes. Therefore, the columns were renamed promptly.
![carbon(1)](https://user-images.githubusercontent.com/74638365/138602907-13486a80-5852-4876-b14f-b5d474dca5cd.png)


```{r}
library(DataExplorer)

plot_intro(df)
```
![image](https://user-images.githubusercontent.com/74638365/138602973-1f03452f-f5fe-4e7e-bb54-d78531f9295a.png)

<br/><br/>

```{r}
# Missing values in the dataset
complete.cases(df)
```
![2](https://user-images.githubusercontent.com/74638365/138603005-4fe405cf-373e-49c4-aad9-6ba14084aed4.PNG)


<br/><br/>

```{r}
# Identify duplicate observations
library(janitor)

get_dupes(df)
```
[1] No duplicate combinations found of: Cultivar, Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, ... and 5 other variables


<br/><br/>

## Exploratory Data Analysis

![carbon(2)](https://user-images.githubusercontent.com/74638365/138603077-0de91fa1-1a67-4f45-a392-e07868ffe253.png)
![image](https://user-images.githubusercontent.com/74638365/138603091-9c6e89ac-9316-48ed-b398-330b607c5e42.png)


<br/><br/>

![carbon(3)](https://user-images.githubusercontent.com/74638365/138603127-b654447c-60b3-4468-8a33-e23ab03ea848.png)
![image](https://user-images.githubusercontent.com/74638365/138603137-f0c8212a-a664-454e-9f70-c3efed05300e.png)

Noticeably strong linear relationship between "Flavanoids" and "Total Phenols"

```{r}
plot(df$Flavanoids, df$`Total phenols`,
     xlab = "Flavanoids",
     ylab = "Total Phenols")
```
![image](https://user-images.githubusercontent.com/74638365/138603232-1122630c-a1c4-469f-8d93-57da66ffb2cc.png)

<br/><br/>

![carbon(4)](https://user-images.githubusercontent.com/74638365/138607848-f5cd647a-dce8-47bc-9cb1-f53168c39e85.png)
![image](https://user-images.githubusercontent.com/74638365/138603305-76c5a9bd-fbf6-49f4-8fe6-6eaef4a55855.png)



## Data Anlysis
### Decision Tree
```{r}
# 75-25 split for random (sample()) training and testing set

set.seed(2021)
division = sort(sample(nrow(df), nrow(df) * 0.75))

training = df[division,]
testing = df[-division,]
```

```{r}
dim(training)
dim(testing)
```

[1] 133 &nbsp; &nbsp;   14

[1] 45  &nbsp; &nbsp; &nbsp;  14

<br/><br/>

![carbon(5)](https://user-images.githubusercontent.com/74638365/138607941-50d78c2f-b2e3-41de-8305-3099a1979668.png)
![3](https://user-images.githubusercontent.com/74638365/138607959-9012971a-b7c8-47a7-85bd-c2d0f6238246.PNG)
![image](https://user-images.githubusercontent.com/74638365/138607968-c810c22f-1b2d-4cc3-be6f-d955c54953b1.png)

<br/><br/>

```{r}
printcp(Dtree)
```

```{r}
plotcp(Dtree)
```
![4](https://user-images.githubusercontent.com/74638365/138607985-14718180-aa8b-4d85-a258-b2efc3d3ae20.PNG)
![image](https://user-images.githubusercontent.com/74638365/138607994-4112c9e8-d90a-4656-8b55-f79480ddeb43.png)



![carbon(6)](https://user-images.githubusercontent.com/74638365/138608008-eeb52aff-14ee-46ef-be56-28aadca8b9a5.png)
![image](https://user-images.githubusercontent.com/74638365/138608017-a8de0f0a-8b6c-4c25-b16f-e1b2cbd07a2a.png)

<br/><br/>

```{r}
prediction = predict(Dtree, testing, type = "class")

confusion_mat = table(testing$Cultivar, prediction)
confusion_mat
```

. | 1 | 2 | 3 |
--|---|---|---|
1 | 13 | 0| 0 |
2 | 1 | 18| 3 |
3 | 0 | 0 | 10|

```{r}
acc = sum(diag(confusion_mat)) / sum(confusion_mat)
print(paste("The accruacy for test is", acc))
```
**[1] "The accruacy for test is 0.911111111111111"**

<br/><br/>

## Hyperparameter tuning

![carbon(7)](https://user-images.githubusercontent.com/74638365/138608248-c36df4b7-afef-4323-b405-e08e88ad2914.png)
![5](https://user-images.githubusercontent.com/74638365/138608276-4ee6a9de-cfdb-42d1-a6a4-ab19ba069a91.PNG)

<br/><br/>

![carbon(8)](https://user-images.githubusercontent.com/74638365/138608307-483ff4c3-1ec1-4a27-837c-08ec949e0ef2.png)
![6](https://user-images.githubusercontent.com/74638365/138608330-5852e6ed-3afd-4635-b1e0-b7191f05918f.PNG)

<br/><br/>

![carbon(9)](https://user-images.githubusercontent.com/74638365/138608380-39312556-91ad-413e-936c-87c8390075b7.png)
![7](https://user-images.githubusercontent.com/74638365/138608416-e1718455-c561-477e-8b3c-a3d5b664cf4d.PNG)

<br/><br/>

```{r}
gridsearch = gridsearch %>% arrange(desc(acc), desc())
gridsearch
```
![8](https://user-images.githubusercontent.com/74638365/138608445-94d21e51-f271-4592-b3ba-0fe17ea012e9.PNG)

<br/><br/>
```{r}
# Selecting the set of hyperparameters with the highest accuracy and minsplit
prp(gridsearch$fit[[4]])
```
![image](https://user-images.githubusercontent.com/74638365/138608476-cfebf559-f305-49c4-9c63-bb027ceab050.png)



![carbon(10)](https://user-images.githubusercontent.com/74638365/138608488-e3bb386a-c3bd-4a72-84b5-686d3275b3af.png)

. | 1 | 2 | 3 |
--|---|---|---|
1 | 13 | 0| 0 |
2 | 0 | 19| 3 |
3 | 0 | 0 | 10|



```{r}
acc2 = sum(diag(confusion_mat2)) / sum(confusion_mat2)
print(paste("The accruacy for test is", acc2))
```

**[1] "The accruacy for test is 0.933333333333333"**


# Conclusion


