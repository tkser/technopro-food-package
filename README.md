# TechnoPro Design Inc. Food Package Image Analysis Challenge (general category/student category)

[Competition Page](https://signate.jp/competitions/1106)<br/>
[Repository](https://github.com/tkser/techpro-food-package)

~ 2023/09/29 23:59:59 (JST)

## Data

### train.csv

|Header Name|Data Type|Description|
|--|--|--|
|image_name|str|Image File Name|
|label|int64|Image Label (0: Beverage, 1: Food)|

### sample_submit.csv

|Header Name|Data Type|Description|
|--|--|--|
|0|str|Image File Name|
|1|float64|Predicted Probability|

### train.zip & test.zip

`*.png` Image Data used for Learning

## Usage

### Preparation
```bash
rye install signate
rye sync
signate download --competition-id=1106 --path=./src/data/input
rye run prepare
```

### Building Model & Prediction
```bash
rye run predict
```

### Submit
```bash
signate submit --competition-id=1106 ./src/data/output/***.csv --note="***"
```