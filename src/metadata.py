
import numpy as np
import pandas as pd

class Metadata():
    def __init__(self, file_path):
        self.df_metadata = pd.read_csv(file_path)
        dict_features_extra = {
            'numeric': [
                         'ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3', 'ALTER_KIND4', 'ANZ_STATISTISCHE_HAUSHALTE',
                         'EINGEFUEGT_AM', 'EINGEZOGENAM_HH_JAHR', 'EXTSEL992', 'VERDICHTUNGSRAUM'
            ],
            'nominal': [
                        'AKT_DAT_KL', 'ALTERSKATEGORIE_FEIN', 'ANZ_KINDER', 'ARBEIT', 'CJT_KATALOGNUTZER',
                        'CJT_TYP_1', 'CJT_TYP_2', 'CJT_TYP_3', 'CJT_TYP_4', 'CJT_TYP_5', 'CJT_TYP_6',
                        'D19_BUCH_CD', 'D19_KONSUMTYP_MAX',
                        'D19_LETZTER_KAUF_BRANCHE', 'D19_SOZIALES',
                        'D19_TELKO_ONLINE_QUOTE_12', 'D19_VERSI_DATUM',
                        'D19_VERSI_OFFLINE_DATUM', 'D19_VERSI_ONLINE_DATUM',
                        'D19_VERSI_ONLINE_QUOTE_12', 'DSL_FLAG', 'FIRMENDICHTE', 'GEMEINDETYP',
                        'HH_DELTA_FLAG', 'KOMBIALTER', 'KONSUMZELLE', 'MOBI_RASTER',
                        'RT_KEIN_ANREIZ', 'RT_SCHNAEPPCHEN', 'RT_UEBERGROESSE',
                        'STRUKTURTYP', 'UMFELD_ALT', 'UMFELD_JUNG', 'UNGLEICHENN_FLAG',
                        'VHA', 'VHN', 'VK_DHT4A', 'VK_DISTANZ', 'VK_ZG11'            
            ],
            'ordinal': [
                        'KBA13_ANTG1', 'KBA13_ANTG2', 'KBA13_ANTG3',
                        'KBA13_ANTG4', 'KBA13_BAUMAX', 'KBA13_GBZ', 'KBA13_HHZ',
                        'KBA13_KMH_210'
            ]
        }
        self.df_lookup = self.df_metadata[['attribute', 'type']]\
                        .drop_duplicates(subset=['attribute', 'type'])\
                        .reset_index(drop=True)
        self.df_lookup['dataset'] = 'metadata'
        
        for key in dict_features_extra.keys():
            n = len(dict_features_extra[key])
            
            self.df_lookup = pd.concat(
                [
                    self.df_lookup,
                    pd.DataFrame({
                                'attribute': dict_features_extra[key],
                                'type': [key] * n,
                                'dataset': ['extra'] * n
                                 })
                ],
                ignore_index=True
            )

    def drop_unknown_values(self):
        # Process: Drop unknown values from metadta frame, as all invalid values not found in metadata will be replaced by np.nan
        self.df_metadata.drop(self.df_metadata.query("meaning == 'unknown'").index, inplace=True)

    def lookup_features(self, input_df, method='intersect', subsets=['metadata', 'extra'], types=['numeric', 'ordinal','nominal']):
        lookup_cols = self.get_features(subsets=subsets, types=types)
        if method == 'intersect':
            func = np.intersect1d
            set_order = (input_df.columns, lookup_cols)
        if method == 'diff':
            func = np.setdiff1d
            set_order = (input_df.columns, lookup_cols)
        if method == 'complement':
            func = np.setdiff1d
            set_order = (lookup_cols, input_df.columns)
        return func(*set_order).tolist()
    
    def get_features(self, subsets=['metadata', 'extra'], types=['numeric', 'ordinal','nominal']):
        lookup_cols = self.df_lookup.query(f"type.isin({types}) and dataset.isin({subsets})", engine="python")\
                          .attribute.tolist()
        return lookup_cols
    
    def get_valid_ranges(self, col_name, col_dtype):
        return self.df_metadata.query(f"attribute == '{col_name}'")['value'].to_numpy().astype(col_dtype)
