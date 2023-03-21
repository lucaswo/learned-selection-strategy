import pandas as pd

# the measurements needs a settingIdx (specific bw histogram), a format (algorithm), 
# and a settingGroup (only required if you want to group settings)
# countValuesSmall should detail the block size
def prepare_data(raw_data: pd.DataFrame, bw_cols: list[str]) -> pd.DataFrame:
    data = raw_data.groupby(["settingIdx", "format", "settingGroup"], as_index=False).mean()

    data.loc[:,bw_cols] = data.loc[:,bw_cols].div(data["countValuesSmall"], axis=0)
    
    data["compressed size [byte]"] = data["compressed size [byte]"]/data["countValuesSmall"]*8

    data.insert(13, "minBucket", data.loc[:,bw_cols].ne(0).idxmax(axis=1).str.replace("bwHist_", "").astype(float).values)
    data.insert(14, "maxBucket", data["bitwidth"].values)
    data.insert(15, "Avg", data[bw_cols].dot(range(1,65)).values)
    data.insert(16, "numBuckets", data.loc[:,bw_cols].ne(0).sum(axis=1).astype(float).values)
    data.insert(17, "Std", data[bw_cols].std(axis=1).values)
    data.insert(18, "Skew", data[bw_cols].skew(axis=1).values)
    data.insert(19, "Kurt", data[bw_cols].kurtosis(axis=1).values)
    
    offset = (data.columns == bw_cols[0]).argmax() - 1
    data["Min"] = data.values[range(len(data)),data["minBucket"].astype(int)+offset]
    data["Max"] = data.values[range(len(data)),data["maxBucket"].astype(int)+offset]
    
    return data
