import streamlit as st

import pandas as pd 
import plotly.express as px
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from datetime import datetime

st.set_page_config(layout="wide")

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def intro():

    image_path = 'data/BGorgeous.png'

    st.write("# Welcome to BGorgeous Demos! 👋")
    st.image(image_path, caption='Your Image Caption', use_column_width=True)
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """

        BGorgeous Demos
    """
    )

def business_dashboard():
    st.header("GT Kovilambakkam Business Dashboard", divider='violet')
    st.markdown("Data Refresh Date: 14 November 2023")

    tickets_df = pd.read_csv("data/Tickets_14Nov23_4pm.csv", encoding='ISO-8859-1', low_memory=False)
    tickets_details_df = pd.read_csv("data/Ticket_Product_Details_14Nov23_4pm.csv", encoding='ISO-8859-1', 
            low_memory=False)

    clients_df = pd.read_csv("data/Client_14Nov23_4pm.csv", encoding='ISO-8859-1', low_memory=False)
    clients_df.dropna(subset=['ClientID'], inplace=True)
    exclude_values = ['.8220484146', '.8393863665', '.9894384197', 'C Balachander9884886817', '0', '..8220484146']
    clients_df = clients_df[~clients_df['ClientID'].isin(exclude_values)]


    tickets_df['Total'] = tickets_df['Total'].fillna(0)
    total_sales = sum(tickets_df['Total'])


    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sales", f"₹{total_sales:,.2f}")

    try:
        tickets_details_df['TicketID'] = tickets_details_df['TicketID'].astype('int64')
        tickets_df['TicketID'] = tickets_df['TicketID'].astype('int64')

    except ValueError:
        tickets_details_df['TicketID'] = pd.to_numeric(tickets_details_df['TicketID'], errors='coerce')
        tickets_df['TicketID'] = pd.to_numeric(tickets_df['TicketID'], errors='coerce')

    services = tickets_details_df[tickets_details_df['Type'] == "S"]
    total_services = services['TicketID'].count()
    col2.metric("Total Services Provided", f"{total_services:,}")

    products = tickets_details_df[tickets_details_df['Type'] == "P"]
    total_products = products['TicketID'].count()
    col3.metric("Total Products Sold", f"{total_products:,}")

    total_customers = len(clients_df['ClientID'].unique())
    col4.metric("Total Customers Served", f"{total_customers:,}")

    

    #Getting Years Extracted
    tickets_df['Bill_Date'] = pd.to_datetime(tickets_df['Created_Date'])
    unique_years = tickets_df['Bill_Date'].dt.year.unique()
    year_list = [x for x in unique_years if not math.isnan(x)]
    year_list = [int(x) for x in year_list]
    

    tickets_details_df['Bill_DateTime'] = pd.to_datetime(tickets_details_df['Start_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    tickets_details_df['Bill_Date'] = tickets_details_df['Bill_DateTime'].dt.date
    tickets_details_df['Bill_Time'] = tickets_details_df['Bill_DateTime'].dt.time
    tickets_details_df = tickets_details_df.drop(columns=['Bill_DateTime', 'Start_Time'])
    tickets_df['Bill_Month'] = tickets_df['Bill_Date'].dt.strftime('%B')

    st.session_state.tickets_details_df = tickets_details_df
    st.session_state.tickets_df = tickets_df
    st.session_state.clients_df = clients_df

    box1, box2 = st.columns(2)
    with box1:
        selected_year = st.selectbox('Select Year', year_list)
    with box2:
        aggregate = st.selectbox("Select Aggregation:", ['Day', 'Month'])

    chart1, chart2, chart3 = st.columns(3)

    with chart1:
        tickets_filt = tickets_df[tickets_df['Bill_Date'].dt.year == selected_year]
        if aggregate == 'Day':
            title = f"Daily Sales Made in {selected_year}"
            fig = px.bar(
                    tickets_filt, 
                    x = 'Bill_Date', 
                    y = 'Total', 
                    title = title, 
                    color_discrete_sequence = ["#8633de"])
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            total_revenue = tickets_filt['Total'].sum()
                # Find the index of the row with the maximum value in the 'Salary' column
            max_sales_index = tickets_filt['Total'].idxmax()
            biggest_date = tickets_filt.loc[max_sales_index, 'Bill_Date']
            biggest_date = datetime.strptime(str(biggest_date), "%Y-%m-%d %H:%M:%S")
            biggest_date = biggest_date.strftime("%B %d, %Y")
        
        if aggregate == 'Month':
            title = f"Monthly Sales Made in {selected_year}"
            fig = px.bar(
                    tickets_filt, 
                    x = 'Bill_Month', 
                    y = 'Total', 
                    title = title, 
                    color_discrete_sequence = ["#8633de"])
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            total_revenue = tickets_filt['Total'].sum()
                # Find the index of the row with the maximum value in the 'Salary' column
            max_sales_index = tickets_filt['Total'].idxmax()
            biggest_date = tickets_filt.loc[max_sales_index, 'Bill_Date']
            biggest_date = datetime.strptime(str(biggest_date), "%Y-%m-%d %H:%M:%S")
            biggest_date = biggest_date.strftime("%B %d, %Y")

                
        st.markdown(
        "You made a total revenue of **{}** in {}. Your biggest day of sales was on **{}**.".format(
            f"₹{total_revenue:,.2f}", selected_year, biggest_date
        )
    )


    with chart2:
        if aggregate == 'Day':
            title = f"Daily Services Provided in {selected_year}"
            services = tickets_details_df[tickets_details_df['Type'] == 'S']
            services['Bill_Date'] = pd.to_datetime(services['Bill_Date'])
            #services['Bill_Date'] = pd.to_datetime(services['Start_Time'], format='%d-%m-%Y')
            services_filtered = services[services['Bill_Date'].dt.year == selected_year]
            fig = px.histogram(services_filtered, x='Bill_Date', title=title, color_discrete_sequence = ["#8633de"])
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            st.markdown(
            "Most popular services this month were..."
            #.format(f"₹{total_revenue:,.2f}", selected_year, biggest_date
        # )
        )

        if aggregate == 'Month':
            title = f"Monthly Services Provided in {selected_year}"
            services = tickets_details_df[tickets_details_df['Type'] == 'S']
            services['Bill_Date'] = pd.to_datetime(services['Bill_Date'])
            services['Bill_Month'] = services['Bill_Date'].dt.strftime('%B')
            #services['Bill_Date'] = pd.to_datetime(services['Start_Time'], format='%d-%m-%Y')
            services_filtered = services[services['Bill_Date'].dt.year == selected_year]
            fig = px.histogram(services_filtered, x='Bill_Month', title=title, color_discrete_sequence = ["#8633de"])
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            st.markdown(
            "Most popular services this month were..."
            #.format(f"₹{total_revenue:,.2f}", selected_year, biggest_date
        # )
        )
    
    with chart3:
        if aggregate == 'Day':
            title = f"Daily Products Sold in {selected_year}"
            products = tickets_details_df[tickets_details_df['Type'] == 'P']
            #products['Bill_Date'] = pd.to_datetime(products['Bill_Date'], format='%d-%m-%Y')
            products['Bill_Date'] = pd.to_datetime(products['Bill_Date'])
            products_filtered = products[products['Bill_Date'].dt.year == selected_year]
            fig = px.histogram(products_filtered, x='Bill_Date', title=title, color_discrete_sequence = ["#8633de"])
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            st.markdown(
            "Most popular products this month were..."
            #.format(f"₹{total_revenue:,.2f}", selected_year, biggest_date
        # )
        )

        if aggregate == 'Month':
            title = f"Monthly Products Sold in {selected_year}"
            products = tickets_details_df[tickets_details_df['Type'] == 'P']
            #products['Bill_Date'] = pd.to_datetime(products['Bill_Date'], format='%d-%m-%Y')
            products['Bill_Date'] = pd.to_datetime(products['Bill_Date'])
            products['Bill_Month'] = products['Bill_Date'].dt.strftime('%B')
            products_filtered = products[products['Bill_Date'].dt.year == selected_year]
            fig = px.histogram(products_filtered, x='Bill_Date', title=title, color_discrete_sequence = ["#8633de"])
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            st.markdown(
            "Most popular products this month were..."
            #.format(f"₹{total_revenue:,.2f}", selected_year, biggest_date
        # )
        )

    
    # Performing an inner merge based on the common column 'ticket_id'
    tickets_details_df['NumServices'] = tickets_details_df.groupby('TicketID')['TicketID'].transform('count')
    tickets_details_df_subset = tickets_details_df[['TicketID', 'NumServices']]
    tickets_df_subset = tickets_df[['TicketID', 'ClientID', 'Total']]
    sales_df = pd.merge(tickets_details_df_subset, tickets_df_subset, on='TicketID', how='right')
    sales_df = sales_df.drop_duplicates(subset=['TicketID'])

    sales_df = sales_df.groupby('ClientID').agg({'NumServices': 'sum', 'Total': 'sum'}).reset_index()
    sales_df.rename(columns={'Total': 'total_spending'}, inplace=True)
    sales_df = pd.merge(sales_df, clients_df, on='ClientID', how='left')

    cuscol1, cuscol2 = st.columns(2)

    with cuscol1:
        # Selecting specific columns from each DataFrame
        clients_df_subset = clients_df[['ClientID', 'Sex']]
        tickets_details_df_subset = tickets_details_df[['TicketID', 'NumServices', 'Bill_Date']]
        tickets_df_subset = tickets_df[['TicketID', 'ClientID', 'Total']]

        merged_df = pd.merge(tickets_details_df_subset, tickets_df_subset, on='TicketID', how='left')
        client_services_df = pd.merge(merged_df, clients_df_subset, on='ClientID', how='left')

        client_services_df['Sex'] = client_services_df['Sex'].apply(lambda x: str(x).upper())
        client_services_df['Sex'] = client_services_df['Sex'].replace("NAN", np.nan)
        client_services_df_filtered = client_services_df[client_services_df['Sex'].notna()]
        #client_services_df['Bill_Date'] = pd.to_datetime(client_services_df['Bill_Date'], format='%d-%m-%Y')
        client_services_df['Bill_Date'] = pd.to_datetime(client_services_df['Bill_Date'])
        client_services_df_filtered = client_services_df[client_services_df['Bill_Date'].dt.year == selected_year]
        grouped = client_services_df_filtered.groupby('Sex')['Total'].mean()
        gender_labels = {
        "F": 'Female',
        "M": 'Male'
    }
        df = pd.DataFrame({
            'Gender': [gender_labels[x] for x in grouped.index],
            'Median Spend': grouped.values
        })

        fig = px.bar(df, x='Median Spend', y='Gender', orientation='h', 
                    labels={'Median Spend': 'Median Spend', 'Gender': 'Gender'},
                    title=f"AVB by Gender in {selected_year}", color_discrete_sequence = ["#8633de"])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    with cuscol2:
        clients_df['Create_Date'] = pd.to_datetime(clients_df['Create_Date'])
        clients_filt = clients_df[clients_df['Create_Date'].dt.year == selected_year]

        clients = pd.DataFrame(clients_filt["category"].dropna().value_counts()).reset_index()
        clients.columns = ["Category", "Count"]

        title = f"Customer Types in {selected_year}"
        fig = px.pie(clients, values='Count', names='Category', title=title, color_discrete_sequence = ["#8633de"])

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    

    



def rfm_model():

        tickets_details_df = st.session_state.tickets_details_df.copy()
        tickets_df = st.session_state.tickets_df.copy()
        clients_df = st.session_state.clients_df.copy()

        st.subheader("Customer Segmentation and RFM Analysis")
        st.markdown("Clustering of customers based on RFM values and scoring customers based on the combined RFM score")

        tickets_details_df['NumServices'] = tickets_details_df.groupby('TicketID')['TicketID'].transform('count')
        df3_subset = tickets_details_df[['TicketID', 'NumServices']]
        df4_subset = tickets_df[['TicketID', 'ClientID', 'Total', 'Bill_Date']]

        sales_df = pd.merge(df3_subset, df4_subset, on='TicketID', how='right')
        sales_df = sales_df.drop_duplicates(subset=['TicketID'])
        sales_df = sales_df[sales_df['ClientID'] != "0"]
        
        cutoff_date = tickets_df['Bill_Date'].max()
        sales_df['LastVisit'] = (cutoff_date - sales_df.groupby('ClientID')['Bill_Date'].transform('max')).dt.days
        sales_df = sales_df.groupby('ClientID').agg({'NumServices': 'sum', 'Total': 'sum', 'LastVisit': 'max'}).reset_index()
        sales_df.rename(columns={'Total': 'TotalSpend'}, inplace=True)
        sales_df.rename(columns={'TotalSpend': 'Monetary', 'NumServices': 'Frequency', 'LastVisit': 'Recency'}, inplace=True)

    

        customers_fix = pd.DataFrame()
        customers_fix["Recency"] = pd.Series(np.cbrt(sales_df['Recency'])).values
        customers_fix["Frequency"] = stats.boxcox(sales_df['Frequency'])[0]
        customers_fix["Monetary"] = pd.Series(np.cbrt(sales_df['Monetary'])).values
        customers_fix.tail()


        scaler = StandardScaler()
        scaler.fit(customers_fix)
        customers_normalized = scaler.transform(customers_fix)


        model = KMeans(n_clusters=3, random_state=42)
        model.fit(customers_normalized)

        sales_df["Cluster"] = model.labels_
        sales_df.loc[:, "Cluster Segment"] = ""
        sales_df.loc[sales_df.loc[:, "Cluster"] == 2, "Cluster Segment"] = "At Risk Customers"
        sales_df.loc[sales_df.loc[:, "Cluster"] == 1, "Cluster Segment"] = "Lost/Churned Customers"
        sales_df.loc[sales_df.loc[:, "Cluster"] == 0, "Cluster Segment"] = "New Customers"

        df_normalized = pd.DataFrame(customers_normalized, columns=['Recency', 'Frequency', 'Monetary'])
        df_normalized['ID'] = sales_df.index
        df_normalized['Cluster'] = model.labels_
        df_normalized['Cluster Segment'] = sales_df['Cluster Segment']

        

        df_nor_melt = pd.melt(df_normalized.reset_index(),
                        id_vars=['ID', 'Cluster Segment'],
                        value_vars=['Recency','Frequency','Monetary'],
                        var_name='Attribute',
                        value_name='Value')

        quantiles = sales_df[['Recency', 'Frequency', 'Monetary', 'Cluster']].quantile(q=[0.2,0.4,0.6,0.8])
        quantiles = quantiles.to_dict()
            
        def RScore(x,p,d):
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
            
        def FMScore(x,p,d):
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
            
        sales_df['R'] = sales_df['Recency'].apply(RScore, args=('Recency',quantiles,))
        sales_df['F'] = sales_df['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
        sales_df['M'] = sales_df['Monetary'].apply(FMScore, args=('Monetary',quantiles,))

        # Concat RFM quartile values to create RFM Segments
        def join_rfm(x): 
            return str(x['R']) + str(x['F']) + str(x['M'])
            
        sales_df['RFM_Segment'] = sales_df.apply(join_rfm, axis=1)
        # Calculate RFM_Score
        sales_df['RFM_Score'] = sales_df[['R','F','M']].sum(axis=1)

        # Create human friendly RFM labels
        segt_map = {
            
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
        # rfm['Segment'] = rfm['R'].map(str) + rfm['F'].map(str)+ rfm['M'].map(str)
        sales_df['Segment'] = sales_df['R'].map(str) + sales_df['F'].map(str)
        sales_df['Segment'] = sales_df['Segment'].replace(segt_map, regex=True)
        # Create some human friendly labels for the scores
        sales_df['Score'] = 'Green'
        sales_df.loc[sales_df['RFM_Score']>5,'Score'] = 'Bronze' 
        sales_df.loc[sales_df['RFM_Score']>7,'Score'] = 'Silver' 
        sales_df.loc[sales_df['RFM_Score']>9,'Score'] = 'Gold' 
        sales_df.loc[sales_df['RFM_Score']>10,'Score'] = 'Platinum'


        value1, value2 = st.columns(2)

        with value1:
            # Aggregate data by each customer
            fig3 = df_nor_melt.groupby('Cluster Segment').agg({'ID': lambda x: len(x)}).reset_index()
            fig3.rename(columns={'ID': 'Count'}, inplace=True)
            fig3['percent'] = (fig3['Count'] / fig3['Count'].sum()) * 100
            fig3['percent'] = fig3['percent'].round(1)

            colors=['#b082f5','#825eb8','#8c42fc'] #color palette

            fig = px.treemap(fig3, path=['Cluster Segment'],values='Count'
                            , width=800, height=550
                            ,title="AI Generated Clusters from RFM Values")

            fig.update_layout(
                treemapcolorway = colors, #defines the colors in the treemap
                margin = dict(t=50, l=25, r=25, b=25))

            fig.data[0].textinfo = 'label+text+value+percent root'
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        with value2:
            # Aggregate data by each customer
            fig4 = sales_df.groupby('Segment').agg({'ClientID': lambda x: len(x)}).reset_index()

            # Rename columns
            fig4.rename(columns={'ClientID': 'Count'}, inplace=True)
            fig4['percent'] = (fig4['Count'] / fig4['Count'].sum()) * 100
            fig4['percent'] = fig4['percent'].round(1)


            colors=['#713ebd','#9771d1','#7d5cad','#a386cf','#6f1aed','#7b38e0','#c2aae6','#6b0ec2'] #color palette

            fig = px.treemap(fig4, path=['Segment'],values='Count'
                            , width=800, height=550
                            ,title="RFM Segments")

            fig.update_layout(
                treemapcolorway = colors, #defines the colors in the treemap
                margin = dict(t=50, l=25, r=25, b=25))

            fig.data[0].textinfo = 'label+text+value+percent root'
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)


        col1, col2 = st.columns(2)
        clients_subset = clients_df[['ClientID', 'FirstName', 'LastName', 'HomePhone', 'Sex', 'DOB', 'survey', 'MembershipCardNo',
                            'Membership_Date']]
        with col1:
            fig5 = sales_df.groupby('Score').agg({'ClientID': lambda x: len(x)}).reset_index()

            # Rename columns
            fig5.rename(columns={'ClientID': 'Count'}, inplace=True)
            fig5['percent'] = (fig5['Count'] / fig5['Count'].sum()) * 100
            fig5['percent'] = fig5['percent'].round(1)

            colors=['#613787','#9d81b8','#7717d1','#7d6296', '#8400ff'] #color palette

            fig = px.treemap(fig5, path=['Score'],values='Count'
                            , width=800, height=800
                            ,title="Customers segmented by RFM Scores")

            fig.update_layout(
                treemapcolorway = colors, #defines the colors in the treemap
                margin = dict(t=50, l=25, r=25, b=25))

            fig.data[0].textinfo = 'label+text+value+percent root'
            st.plotly_chart(fig, theme='streamlit', use_container_width=True)

        desired_columns = ['ClientID', 'Cluster Segment', 'FirstName', 'LastName', 'Days since last visit', 
                                        'Number of Visits', 'Total Billed', 'Segment', 'Score', 'HomePhone',
                                'Sex', 'DOB', 'survey', 'MembershipCardNo', 'Membership_Date']
        
        with col2:
            st.markdown("**Gold Customers**")
            gold_customers = sales_df[sales_df['Score'] == "Gold"]
            gold_customers = pd.merge(gold_customers, clients_subset, on='ClientID', how='left')
            gold_customers = gold_customers.rename(columns={'Recency': 'Days since last visit', 'Frequency': 'Number of Visits',
                         'Monetary': 'Total Billed'})
            gold_customers = gold_customers[desired_columns]
            st.dataframe(gold_customers.sample(3))
            csv = convert_df(gold_customers)
            st.download_button(
                label="Download gold customer data",
                data=csv,
                file_name='gold_customers.csv',
                mime='text/csv', use_container_width=True
            )

            st.markdown("**Platinum Customers**")
            platinum_customers = sales_df[sales_df['Score'] == "Platinum"]
            platinum_customers = pd.merge(platinum_customers, clients_subset, on='ClientID', how='left')
            platinum_customers = platinum_customers.rename(columns={'Recency': 'Days since last visit', 'Frequency': 'Number of Visits',
                                    'Monetary': 'Total Billed'})
            platinum_customers = platinum_customers[desired_columns]
            st.dataframe(platinum_customers.sample(3))
            csv = convert_df(platinum_customers)
            st.download_button(
                label="Download platinum customer data",
                data=csv,
                file_name='platinum_customers.csv',
                mime='text/csv', use_container_width=True
            )

            st.markdown("**Green Customers**")
            green_customers = sales_df[sales_df['Score'] == "Green"]
            green_customers = pd.merge(green_customers, clients_subset, on='ClientID', how='left')
            green_customers = green_customers.rename(columns={'Recency': 'Days since last visit', 'Frequency': 'Number of Visits',
                            'Monetary': 'Total Billed'})
            green_customers = green_customers[desired_columns]
            st.dataframe(green_customers.sample(3))
            csv = convert_df(green_customers)
            st.download_button(
                label="Download green customer data",
                data=csv,
                file_name='green_customers.csv',
                mime='text/csv', use_container_width=True
            )

            st.markdown("**Bronze Customers**")
            bronze_customers = sales_df[sales_df['Score'] == "Bronze"]
            bronze_customers = pd.merge(bronze_customers, clients_subset, on='ClientID', how='left')
            bronze_customers = bronze_customers.rename(columns={'Recency': 'Days since last visit', 'Frequency': 'Number of Visits',
                    'Monetary': 'Total Billed'})
            bronze_customers = bronze_customers[desired_columns]
            st.dataframe(bronze_customers.sample(3))
            csv = convert_df(bronze_customers)
            st.download_button(
                label="Download bronze customer data",
                data=csv,
                file_name='bronze_customers.csv',
                mime='text/csv', use_container_width=True
            )

            st.markdown("**Silver Customers**")
            silver_customers = sales_df[sales_df['Score'] == "Silver"]
            silver_customers = pd.merge(silver_customers, clients_subset, on='ClientID', how='left')
            silver_customers = silver_customers.rename(columns={'Recency': 'Days since last visit', 'Frequency': 'Number of Visits',
                            'Monetary': 'Total Billed'})
            silver_customers = silver_customers[desired_columns]
            st.dataframe(silver_customers.sample(3))
            csv = convert_df(silver_customers)
            st.download_button(
                label="Download silver customer data",
                data=csv,
                file_name='silver_customers.csv',
                mime='text/csv', use_container_width=True
            )

def recommendation_model():

    st.subheader("Recommendation Model")

    
page_names_to_funcs = {
    "—": intro,
    "Business Dashboard Demo": business_dashboard,
    "RFM Model Demo": rfm_model,
    "Recommendation Engine Demo": recommendation_model
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

#if __name__ == "__main__":
   # main()