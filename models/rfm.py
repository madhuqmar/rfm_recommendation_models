import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

class RFMModel():
    def __init__(self, sales_df, user_col, date_col, spend_col):
        self.sales_df = sales_df #bookings_df
        self.user_col = user_col #user_id #ClientID
        self.date_col = date_col #date #CreatedDate
        self.spend_col = spend_col 
        
        # Create human friendly RFM labels
        self.segt_map = {
            
            r'[1-2][1-2]': 'Hibernating',
            r'[1-2][3-4]': 'At risk',
            r'[1-2]5': 'Can\'t lose them',
            r'3[1-2]': 'About to sleep',
            r'33': 'Need attention',
            r'[3-4][4-5]': 'Loyal customers',
            r'41': 'Promising',
            r'51': 'New customers',
            r'[4-5][2-3]': 'Potential loyalists',
            r'5[4-5]': 'Champions'
        }

        self.segment_descriptions = {
                'Hibernating': 'Customers with low recency, frequency, and monetary values. They may need re-engagement.',
                'At risk': 'Customers who have not made purchases recently and are at risk of churning.',
                'Can\'t lose them': 'High-value customers who have not been active recently. High priority for retention.',
                'About to sleep': 'Customers with low engagement who may stop purchasing soon.',
                'Need attention': 'Customers who need attention to prevent them from churning.',
                'Loyal customers': 'Customers who purchase frequently and recently, showing loyalty.',
                'Promising': 'Customers with potential to become loyal if nurtured properly.',
                'New customers': 'Recently acquired customers with the potential for future loyalty.',
                'Potential loyalists': 'Customers showing signs of becoming loyal with a bit more engagement.',
                'Champions': 'Top customers with high recency, frequency, and monetary values. They are the most valuable.'
            }
        
        self.tier_descriptions = {
                'Platinum': 'Top tier customers with the highest RFM scores, representing the most valuable and loyal customers.',
                'Gold': 'High tier customers with strong loyalty and value, just below Platinum.',
                'Silver': 'Mid-tier customers with moderate loyalty and value.',
                'Bronze': 'Lower tier customers who are less engaged or at risk of churning.',
                'Green': 'New or low-engagement customers who haven’t yet shown strong purchasing behavior.'
            }


    def transform_data(self):
        self.sales_df['TotalSpend'] = self.sales_df.groupby([self.user_col])[self.spend_col].transform('sum')
        self.sales_df['NumVisits'] = self.sales_df.groupby([self.user_col])[self.user_col].transform('count')
        self.sales_df[self.date_col] = pd.to_datetime(self.sales_df[self.date_col], errors='coerce')
        cutoff_date = self.sales_df[self.date_col].max()
        self.sales_df['DaysSinceLastVisit'] = (cutoff_date - self.sales_df.groupby(self.user_col)[self.date_col].transform('max')).dt.days

        self.sales_df["Monetary"] = self.sales_df["TotalSpend"]
        self.sales_df["Frequency"] = self.sales_df["NumVisits"]
        self.sales_df["Recency"] =self.sales_df["DaysSinceLastVisit"]

        customers_fix = pd.DataFrame()
        customers_fix["Recency"] = pd.Series(np.cbrt(self.sales_df['Recency'])).values
        customers_fix["Frequency"] = pd.Series(np.cbrt(self.sales_df['Frequency'])).values
        customers_fix["Monetary"] = pd.Series(np.cbrt(self.sales_df['Monetary'])).values

        customers_fix.dropna(inplace=True)

        return customers_fix

    def fit_and_save_model(self, model_path="rfm_model.pkl", scaler_path="rfm_scaler.pkl", quantiles_path="rfm_quantiles.pkl"):
        """ KMeans clustering """

        customers_fix = self.transform_data()
        scaler = StandardScaler().fit(customers_fix)
        joblib.dump(scaler, scaler_path)

        customers_normalized = scaler.transform(customers_fix)

        ## PER ELBOW METHOD ANALYSIS 4 SEEMS THE OPTIMAL NUMBER OF CLUSTERS ##
        model = KMeans(n_clusters=4, random_state=42)
        model.fit(customers_normalized)

        joblib.dump(model, model_path)
        
        quantiles = self.sales_df[['Recency', 'Frequency', 'Monetary']].quantile(q=[0.2,0.4,0.6,0.8]).to_dict()
        joblib.dump(quantiles, quantiles_path)


    def predict_segment(self):
        """ Predict customer segments using pre-trained KMeans model and scaler """
        
        scaler = joblib.load("models/rfm_scaler.pkl")
        model = joblib.load("models/rfm_model.pkl")

        # Drop rows with missing values in key columns
        self.sales_df.dropna(subset=["Recency", "Frequency", "Monetary"], inplace=True)
        self.sales_df.reset_index(drop=True, inplace=True)

        # Select features
        features = self.sales_df[["Recency", "Frequency", "Monetary"]]

        # Guard clause for empty data
        if features.empty:
            raise ValueError("No data available for prediction. Ensure sales_df has non-empty Recency, Frequency, and Monetary columns.")

        # Transform and predict
        features_scaled = scaler.transform(features)
        self.sales_df["Cluster"] = model.predict(features_scaled)

        # Map clusters to segment names
        cluster_map = {
            3: "New Customers",
            2: "At Risk Customers",
            1: "Loyal Customers",
            0: "Churned Customers"
        }
        self.sales_df["Cluster Segment"] = self.sales_df["Cluster"].map(cluster_map)

        return self.sales_df
    
    def RScore(self,x,p,d):
        if x <= d[p][0.2]:
            return 5
        elif x <= d[p][0.4]:
            return 4
        elif x <= d[p][0.6]: 
            return 3
        elif x <= d[p][0.8]: 
            return 2
        else:
            return 1   
            
    def FMScore(self,x,p,d):
        if x <= d[p][0.2]:
            return 1
        elif x <= d[p][0.4]:
            return 2
        elif x <= d[p][0.6]: 
            return 3
        elif x <= d[p][0.8]: 
            return 4
        else:
            return 5
        
            
    def join_rfm(self, x): 
        """ Concat RFM quartile values to create RFM Segments"""
        return str(x['R']) + str(x['F']) + str(x['M'])

    def rf_segmentation(self):
        """ To get RF segmentaion of customers"""

        quantiles = joblib.load("models/rfm_quantiles.pkl")
            
        self.sales_df['R'] = self.sales_df['Recency'].apply(self.RScore, args=('Recency',quantiles,))
        self.sales_df['F'] = self.sales_df['Frequency'].apply(self.FMScore, args=('Frequency',quantiles,))
        self.sales_df['M'] = self.sales_df['Monetary'].apply(self.FMScore, args=('Monetary',quantiles,))

        self.sales_df['RFM_Segment'] = self.sales_df.apply(self.join_rfm, axis=1)
        self.sales_df['Segment'] = self.sales_df['R'].map(str) + self.sales_df['F'].map(str)
        self.sales_df['Segment'] = self.sales_df['Segment'].replace(self.segt_map, regex=True)

        self.sales_df['Description'] = self.sales_df['Segment'].map(self.segment_descriptions)

        return self.sales_df

    def rfm_tiering(self):
        """Add RFM Score and Tier information (Green to Platinum) to sales_df"""

        # Calculate RFM_Score
        self.sales_df['RFM_Score'] = self.sales_df[['R', 'F', 'M']].sum(axis=1)

        # Assign Tier based on score thresholds
        self.sales_df['Tier'] = 'Green'
        self.sales_df.loc[self.sales_df['RFM_Score'] > 5, 'Tier'] = 'Bronze'
        self.sales_df.loc[self.sales_df['RFM_Score'] > 7, 'Tier'] = 'Silver'
        self.sales_df.loc[self.sales_df['RFM_Score'] > 9, 'Tier'] = 'Gold'
        self.sales_df.loc[self.sales_df['RFM_Score'] > 10, 'Tier'] = 'Platinum'

        # Map description
        self.sales_df['description'] = self.sales_df['Tier'].map(self.tier_descriptions)

        return self.sales_df
