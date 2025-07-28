from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

class RecommendationModel:
    def __init__(self, sales_df, sales_services_df, service_details_df, service_cat_df, service_date_col, 
                 booking_id_col, client_id_col, service_id_col, service_cat_id_col, service_name_col):
        self.sales_df = sales_df
        self.sales_services_df = sales_services_df
        self.service_details_df = service_details_df
        self.service_cat_df = service_cat_df

        self.service_date_col = service_date_col
        self.booking_id_col = booking_id_col
        self.client_id_col = client_id_col
        self.service_name_col = service_name_col
        self.service_id_col = service_id_col
        self.service_cat_id_col = service_cat_id_col

        self.sales_df[self.service_date_col] = pd.to_datetime(self.sales_df[self.service_date_col])
        self.sales_services_df[self.service_date_col] = pd.to_datetime(sales_services_df[self.service_date_col])
        self.sales_df[self.booking_id_col] = self.sales_df[self.booking_id_col].astype('int64', errors='ignore')
        self.sales_services_df[self.booking_id_col] = self.sales_services_df[self.booking_id_col].astype('int64', errors='ignore')

        self.sales_df = self.sales_df.rename(columns={'id': self.booking_id_col})
        self.service_details_df = self.service_details_df.rename(columns={'id': self.service_id_col, 'type_id': self.service_cat_id_col})
        self.service_cat_df = self.service_cat_df.rename(columns={'id': self.service_cat_id_col})

    def filter_and_merge_data(self):
        # tickets_details_df = tickets_details_df[~tickets_details_df['Group1'].isna()]

        client_services = pd.merge(self.sales_services_df, self.sales_df, on=self.booking_id_col, how='left')

        client_services = pd.merge(client_services, self.service_details_df, on=self.service_id_col, how='left')
        client_services = pd.merge(client_services, self.service_cat_df, on=[self.service_cat_id_col, 'salon_id'], how='inner')

        client_services['Frequency'] = client_services.groupby([self.client_id_col, self.service_name_col])[self.client_id_col].transform('size')
        client_services.reset_index(inplace=True)

        return client_services

    def create_pivot_table(self, client_services):
        client_services = client_services[[self.client_id_col, self.service_name_col, 'Frequency']]
        all_pivots = client_services.pivot_table(index=self.client_id_col, columns=self.service_name_col, values='Frequency', fill_value=0)
        pivot_df_sample = all_pivots.sample(100, random_state=42)

        return pivot_df_sample
    
    def fit_similarity_model(self, save_path="models/"):
        client_services = self.filter_and_merge_data(self.sales_df, self.sales_services_df)
        pivot_df_sample = self.create_pivot_table(client_services)

        customer_similarity = cosine_similarity(pivot_df_sample)

        # Save both pivot_df and similarity matrix
        os.makedirs(save_path, exist_ok=True)

        pivot_df_sample.to_pickle(os.path.join(save_path, "pivot_df.pkl"))
        np.save(os.path.join(save_path, "customer_similarity.npy"), customer_similarity)

    def recommend_services(self, threshold=0.75, load_path="models/"):

        pivot_df = pd.read_pickle(os.path.join(load_path, "pivot_df.pkl"))
        customer_similarity = np.load(os.path.join(load_path, "customer_similarity.npy"))

        # Generate recommendations
        recommendations = {}
        for idx, customer in enumerate(pivot_df.index):
            similar_customers = sorted(
                list(enumerate(customer_similarity[idx])),
                key=lambda x: x[1],
                reverse=True
            )
            recommendations[customer] = []
            for similar_customer, similarity_score in similar_customers[1:]:  # skip self
                if similarity_score > threshold:
                    similar_services = pivot_df.columns[pivot_df.iloc[similar_customer] > 0].tolist()
                    for service in similar_services:
                        if pivot_df.loc[customer, service] == 0:
                            recommendations[customer].append((service, similarity_score))

            # Remove duplicates by service name
            seen = set()
            recommendations[customer] = [
                (s, score) for s, score in recommendations[customer] if not (s in seen or seen.add(s))
            ]

        # Return only customers who have recommendations
        return {
            cust: recs for cust, recs in recommendations.items() if len(recs) > 0
        }
