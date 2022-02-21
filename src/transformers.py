import re

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder, StandardScaler)

class CustomPreprocessor(BaseEstimator,TransformerMixin):
    def __init__(self, metadata):
        self.metadata = metadata
        
        # Process: Drop unknown values from metadta frame, as all invalid values not found in metadata will be replaced by np.nan
        self.metadata.drop_unknown_values()     

    def replace_invalid_values(self, series):
        series = series.copy()
        col_name = series.name
        col_dtype = series.dtype
        valid_metadata_ranges = self.metadata.get_valid_ranges(col_name, col_dtype)

        if col_dtype == object:
            dataset_ranges = series.fillna('NULL').unique().astype(col_dtype)
        else:
            dataset_ranges = series.unique().astype(col_dtype)

        invalid_values = np.setdiff1d(
            dataset_ranges,
            valid_metadata_ranges
        )
        series.replace(invalid_values, np.nan, inplace=True)

        return series


    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        
        # Process: Remove invalid entries in ['CAMEO_DEUG_2015', 'CAMEO_INTL_2015'] and change dtype to float
        X.iloc[:,18].replace('X', np.nan, inplace=True)
        X.iloc[:,19].replace('XX', np.nan, inplace=True)
        X.iloc[:,[18,19]] = X.iloc[:,[18,19]].astype('float')
        
        # Process: Drop 'LNR' column as it has unique values:
        X.drop('LNR', axis=1, inplace=True)
        
        # Process: Columns that exist in dataset that don't have metadata:
        extra_cols_dataset = self.metadata.lookup_features(X, method='diff', subsets=['metadata'])

        # Process: Columns that exist in metadata and not in the dataset:
        extra_cols_metadata = self.metadata.lookup_features(X, method='complement', subsets=['metadata'])

        # Process: Rename D19 columns in dataset exist in metadata to match metadata, suffix them with _RZ
        X.columns =  X.columns.map(
            lambda x: re.sub(r"^(D19_.*)$", r"\1_RZ", x) 
            if x in np.intersect1d([re.sub(r"_RZ", "", x) for x in extra_cols_metadata], extra_cols_dataset) 
            else x
        )
        
        # Process: Manually associate and rename columns
        X.rename(columns={
                        'CAMEO_INTL_2015':'CAMEO_DEUINTL_2015',
                        'KK_KUNDENTYP': 'D19_KK_KUNDENTYP',
                        'KBA13_CCM_1401_2500': 'KBA13_CCM_1400_2500', 
                        'D19_BUCH_CD_RZ': 'D19_BUCH_RZ',
                        'SOHO_KZ': 'SOHO_FLAG'
                        }, 
                        inplace=True
                )

        # Process: convert `'EINGEFUEGT_AM'` column that is a date to year column.
        X['EINGEFUEGT_AM'] = X['EINGEFUEGT_AM'].astype('datetime64[ns]').dt.year
        
        # Process: find categorical features that exist in both the dataset and metadata.
        dataset_meta_categorical = self.metadata.lookup_features(
            X,
            method='intersect',
            subsets=['metadata'], 
            types=['ordinal', 'nominal']
        )
        
        # Process: find and replace invalid values with np.nan
        X[dataset_meta_categorical] = X[dataset_meta_categorical].apply(self.replace_invalid_values)
               
        return X

class MissingDataColsRemover(BaseEstimator,TransformerMixin):
    def __init__(self, missing_threshold=0.3):
        self.missing_threshold = missing_threshold
        
    def fit(self,X,y=None):
        missing_columns_freqs = (X.isna().sum() / X.shape[0]).sort_values(ascending = False)
        self.missing_data_cols = missing_columns_freqs[missing_columns_freqs > self.missing_threshold].index.tolist()
        return self
    
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        X = X.drop(self.missing_data_cols, axis=1)       
        self.feature_names_out = X.columns.to_list()
        return X
    def get_feature_names_out(self, *args):
        return self.feature_names_out

class TrainMissingDataRowsRemover(TransformerMixin):
    def __init__(self, missing_threshold=0.3):
        self.missing_threshold = missing_threshold

    def fit(self,X,y=None):
        #Inplace transformation of X and Y
        missing_rows_freqs = (X.isna().sum(axis=1) / X.shape[1]).sort_values(ascending = False)
        drop_idx = missing_rows_freqs[missing_rows_freqs > self.missing_threshold].index.tolist()        
        X.drop(drop_idx, inplace=True)
        y.drop(drop_idx, inplace=True)
        return self
    
    def transform(self,X,y=None):
        return X

class CorrelatedRemover(BaseEstimator,TransformerMixin):
    def __init__(self, correlation_threshold=0.6):
        self.correlation_threshold = correlation_threshold
        

    def fit(self,X,y=None):
        correlation_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper_corr_mat = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

        self.correlated_cols_to_drop = upper_corr_mat.columns[(upper_corr_mat > self.correlation_threshold).any(axis='rows')].tolist()

        return self

    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        X.drop(self.correlated_cols_to_drop, axis=1, inplace=True)
        self.feature_names_out = X.columns.to_list()
        return X
    def get_feature_names_out(self, *args):
        return self.feature_names_out

class TrainDuplicatesRemover(TransformerMixin):
    def __init__(self):
        None

    def fit(self,X,y=None):
        #Inplace transformation of X and Y
        drop_idx = X[X.duplicated()].index
        X.drop(drop_idx, inplace=True)
        y.drop(drop_idx, inplace=True)
        return self
    
    def transform(self,X,y=None):
        return X

class CustomSimpleImputer(SimpleImputer):
    def __init__(self, strategy, **kwargs):
        super().__init__(strategy=strategy, **kwargs)
    def transform(self, X):
        output = super().transform(X)
        return pd.DataFrame(output, columns=self.feature_names_in_, index=X.index)
    def get_feature_names_out(self, column):
        return column

# 3 Standard Deviation Outliers
class TrainOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        selector_callable = lambda X: X.columns, 
    ):
        self.selector_callable = selector_callable
        
    def fit(self, X, y):
        query_cols = self.selector_callable(X)
        outliers_idx = X[X[query_cols].apply(self.get_outliers).any(axis=1)].index
        X.drop(outliers_idx, inplace=True)
        y.drop(outliers_idx, inplace=True)
        return self

    def transform(self, X, y=None):
        return X
    
    def get_outliers(self, s):
        mean , std = s.mean(), s.std()
        cutoff = 3 * std 
        return ~ s.between(mean - cutoff, mean + cutoff) # a series that has True in place of outliers

class CustomColumnTransformer(ColumnTransformer):
    def __init__(
        self,
        transformers,
        verbose_feature_names_out=False,
        **kwargs
    ):
        super().__init__(
        transformers = transformers,
        verbose_feature_names_out=False,
        **kwargs
    )
    def fit_transform(self, X, y=None):
        output = super().fit_transform(X, y)
        return pd.DataFrame(
            output if type(output) == np.ndarray else output.toarray(),
            columns=self.get_feature_names_out(),
            index = X.index,
        ) #.convert_dtypes()
    
    def transform(self, X):
        output = super().transform(X)
        return pd.DataFrame(
            output if type(output) == np.ndarray else output.toarray(),
            columns=self.get_feature_names_out(),
            index = X.index,
        ) #.convert_dtypes()

