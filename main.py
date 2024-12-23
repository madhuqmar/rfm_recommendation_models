import streamlit as st
import pandas as pd 
import plotly.express as px
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")

#Change according to the salon 
salon_brand_color = ["#7a1279"]

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


def intro():

    image_path = 'data/BGorgeous.png'

    st.write("# Welcome to BGorgeous Demos! 👋")
    # st.image(image_path, caption='Your Image Caption')
    st.sidebar.success("Select a demo above.")

    exclude_clients = ['.8220484146', '.8393863665', '.9894384197', 'C Balachander9884886817', '0', '..8220484146']
    
    if st.button('Download data before running the demo apps'):
        ## DOWNLOAD DATA ##
        tickets_df = pd.read_csv("data/Tickets_14Nov23_4pm.csv", encoding='ISO-8859-1', low_memory=False)
        tickets_df.dropna(subset=['ClientID'], inplace=True)
        tickets_df = tickets_df[~tickets_df['ClientID'].isin(exclude_clients)]
        tickets_df['Total'] = tickets_df['Total'].fillna(0)
        tickets_df['Bill_Date'] = pd.to_datetime(tickets_df['Created_Date'])
        tickets_df['Bill_Month'] = tickets_df['Bill_Date'].dt.strftime('%B')
        tickets_df['Bill_Date']['Bill_Date'] = pd.to_datetime(tickets_df['Bill_Date'])
        tickets_df['NumVisits'] = tickets_df.groupby(['ClientID'])['ClientID'].transform('count')
        tickets_df['TotalSpend'] = tickets_df.groupby(['ClientID'])['Total'].transform('sum')
        tickets_df['Created_Date'] = pd.to_datetime(tickets_df['Created_Date'])
        cutoff_date = tickets_df['Created_Date'].max()
        tickets_df['DaysSinceLastVisit'] = (cutoff_date - tickets_df.groupby('ClientID')['Created_Date'].transform('max')).dt.days

            
        tickets_details_df = pd.read_excel("data/New_Ticket_Product_Details_14Nov_23.xlsx")

        try:
            tickets_details_df['TicketID'] = tickets_details_df['TicketID'].astype('int64')
            tickets_df['TicketID'] = tickets_df['TicketID'].astype('int64')

        except ValueError:
            tickets_details_df['TicketID'] = pd.to_numeric(tickets_details_df['TicketID'], errors='coerce')
            tickets_df['TicketID'] = pd.to_numeric(tickets_df['TicketID'], errors='coerce')

        tickets_details_df['Bill_DateTime'] = pd.to_datetime(tickets_details_df['Created_Date2'], format='%d-%m-%Y %H:%M', errors='coerce')
        tickets_details_df['Bill_Date'] = tickets_details_df['Bill_DateTime'].dt.date
        tickets_details_df['Bill_Time'] = tickets_details_df['Bill_DateTime'].dt.time
        tickets_details_df = tickets_details_df.drop(columns=['Bill_DateTime', 'Created_Date2'])
        tickets_details_df['Bill_Date'] = pd.to_datetime(tickets_details_df['Bill_Date'])
        tickets_details_df['NumServices'] = tickets_details_df[tickets_details_df['Type'] == 'S'].groupby('TicketID')['TicketID'].transform('count')
        

        clients_df = pd.read_csv("data/Client_14Nov23_4pm.csv", encoding='ISO-8859-1', low_memory=False)
        clients_df.dropna(subset=['ClientID'], inplace=True)
        clients_df = clients_df[~clients_df['ClientID'].isin(exclude_clients)]

        df3_temp = tickets_details_df[['TicketID', 'NumServices', 'Bill_Date']]
        df4_temp = tickets_df[['TicketID', 'ClientID', 'TotalSpend', 'NumVisits', 'TotalSpend', 'DaysSinceLastVisit']]
        merge_temp = pd.merge(df3_temp, df4_temp, on='TicketID', how='left')
        merge_temp.drop_duplicates(inplace = True)

        yearly_services = pd.DataFrame(merge_temp.groupby([merge_temp['ClientID'], merge_temp['Bill_Date'].dt.year])['NumServices'].sum().fillna(0))
        yearly_services['avg_yrly_services'] = yearly_services.groupby('ClientID')['NumServices'].transform('mean')
        yearly_services.reset_index(inplace=True)
        yearly_services.drop(columns=['Bill_Date', 'NumServices'], inplace=True)
        yearly_services.drop_duplicates(inplace=True)

        monthly_services = pd.DataFrame(merge_temp.groupby([merge_temp['ClientID'], merge_temp['Bill_Date'].dt.month])['NumServices'].sum().fillna(0))
        monthly_services['avg_mnthly_services'] = monthly_services.groupby('ClientID')['NumServices'].transform('mean')
        monthly_services.reset_index(inplace=True)
        monthly_services.drop(columns=['Bill_Date', 'NumServices'], inplace=True)
        monthly_services.drop_duplicates(inplace=True)

        client_service_avgs = pd.merge(yearly_services, monthly_services, on='ClientID', how='outer')

        df4_subset = tickets_df[['ClientID', 'TotalSpend', 'NumVisits', 'DaysSinceLastVisit']]
        sales_df = pd.merge(df4_subset, client_service_avgs, on='ClientID', how='left')
        sales_df.drop_duplicates(inplace=True)
        sales_df.dropna(subset=['ClientID'])

        df1_subset = clients_df[['ClientID', 'Sex', 'HomePhone']]
        df1_subset.dropna(subset=['ClientID'], inplace=True)
        sales_df = pd.merge(sales_df, df1_subset, on='ClientID', how='left')
        sales_df['Sex'] = sales_df['Sex'].apply(lambda x: str(x).upper())
        sales_df['Sex'] = sales_df['Sex'].replace("NAN", np.nan)
        
        if 'tickets_details_df' not in st.session_state:
            st.session_state.tickets_details_df = tickets_details_df
        if 'tickets_df' not in st.session_state:
            st.session_state.tickets_df = tickets_df
        if 'clients_df' not in st.session_state:
            st.session_state.clients_df = clients_df
        if 'sales_df' not in st.session_state:
            st.session_state.sales_df = sales_df
        
        st.success("Data downloaded!")

def business_dashboard():
    st.header("GT Kovilambakkam Business Dashboard", divider='grey')
    st.markdown("Data Refresh Date: 14 November 2023")

    tickets_df = st.session_state.tickets_df.copy()
    tickets_details_df = st.session_state.tickets_details_df.copy()
    clients_df = st.session_state.clients_df.copy()
    sales_df = st.session_state.sales_df.copy()


    ## OVERALL ALL TIME BUSINESS METRICS ##
    col1, col2, col3, col4 = st.columns(4)

    total_sales = sum(tickets_df['Total'])
    col1.metric("Total Sales", f"₹{total_sales:,.2f}")

    services = tickets_details_df[tickets_details_df['Type'] == "S"]
    total_services = services['TicketID'].count()
    col2.metric("Total Services Provided", f"{total_services:,}")

    products = tickets_details_df[tickets_details_df['Type'] == "P"]
    total_products = products['TicketID'].count()
    col3.metric("Total Products Sold", f"{total_products:,}")

    total_customers = len(tickets_df['ClientID'].unique())
    col4.metric("Total Customers Served", f"{total_customers:,}")

    
    ## Extracting Available Years ## 
    unique_years = tickets_df['Bill_Date'].dt.year.unique()
    year_list = [x for x in unique_years if not math.isnan(x)]
    year_list = [int(x) for x in year_list]
    

    ## USER SELECTION BOXES ##
    box1, box2 = st.columns(2)
    with box1:
        selected_year = st.selectbox('Select Year', year_list)
    with box2:
        aggregate = st.selectbox("Select Aggregation:", ['Day', 'Month'])

    ## CHARTS ABOUT REVENUE ##
    chart1, chart2, chart3 = st.columns(3)

    with chart1:
        tickets_filt = tickets_df[tickets_df['Bill_Date'].dt.year == selected_year]
        tickets_filt = tickets_filt.groupby('Bill_Date')['Total'].sum().reset_index()
        tickets_filt['Bill_Month'] = tickets_filt['Bill_Date'].dt.strftime('%B')
        tickets_filt_by_month = tickets_filt.groupby('Bill_Month')['Total'].sum().reset_index()

        max_sales_index = tickets_filt['Total'].idxmax()
        biggest_date = tickets_filt.loc[max_sales_index, 'Bill_Date']
        biggest_date = datetime.strptime(str(biggest_date), "%Y-%m-%d %H:%M:%S")
        biggest_date = biggest_date.strftime("%B %d, %Y")
        biggest_day_sales = tickets_filt.loc[max_sales_index, 'Total']

        biggest_month_index = tickets_filt_by_month['Total'].idxmax()
        biggest_month = tickets_filt_by_month.loc[biggest_month_index, 'Bill_Month']
        biggest_month_sales = tickets_filt_by_month.loc[biggest_month_index, 'Total']

        if aggregate == 'Day':
            title = f"Daily Sales Made in {selected_year}"
            fig = px.bar(
                    tickets_filt, 
                    x = 'Bill_Date', 
                    y = 'Total', 
                    title = title, 
                    color_discrete_sequence = salon_brand_color)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            total_revenue = tickets_filt['Total'].sum()
            
        if aggregate == 'Month':
            title = f"Monthly Sales Made in {selected_year}"

            fig = px.bar(
                    tickets_filt, 
                    x = 'Bill_Month', 
                    y = 'Total', 
                    title = title, 
                    color_discrete_sequence = salon_brand_color)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            total_revenue = tickets_filt['Total'].sum()
                # Find the index of the row with the maximum value in the 'Salary' column
            max_sales_index = tickets_filt['Total'].idxmax()
            biggest_date = tickets_filt.loc[max_sales_index, 'Bill_Date']
            biggest_date = datetime.strptime(str(biggest_date), "%Y-%m-%d %H:%M:%S")
            biggest_date = biggest_date.strftime("%B %d, %Y")

            biggest_month_index = tickets_filt_by_month['Total'].idxmax()
            biggest_month = tickets_filt_by_month.loc[biggest_month_index, 'Bill_Month']
   
        st.markdown(
        "You made a total revenue of **{}** in {}. Your biggest day of sales was on **{}** where you made **{}**. Your biggest month of sales was in **{}** when you made **{}**".format(
            f"₹{total_revenue:,.2f}", selected_year, biggest_date, f"₹{ biggest_day_sales:,.2f}", biggest_month, f"₹{biggest_month_sales:,.2f}" 
        )
    )


    with chart2:
        services = tickets_details_df[tickets_details_df['Type'] == 'S']
        services['Bill_Date'] = pd.to_datetime(services['Bill_Date'])
        services['Bill_Month'] = services['Bill_Date'].dt.strftime('%B')
        services_filtered = services[services['Bill_Date'].dt.year == selected_year]
        service_counts = services_filtered.groupby('Descr').agg(
                    frequency=('Descr', 'size'),  # Rename 'size' column to 'frequency'
                    total_sum=('Total', 'sum')
                ).reset_index()
        service_counts = service_counts.sort_values(by='frequency', ascending=False)
        top_services_df = service_counts.head(3)
        top_services = list(top_services_df['Descr'])
        top_services = ', '.join("'" + element + "'" for element in top_services)
        top_services_freq = list(top_services_df['frequency']) 

        top_services_rev = list(top_services_df['total_sum']) 
        top_services_rev  = [f"₹{number:,.2f}" for number in top_services_rev]
        top_services_rev  = ', '.join(top_services_rev)


        if aggregate == 'Day':

            title = f"Daily Services Provided in {selected_year}"
            fig = px.histogram(services_filtered, x='Bill_Date', title=title, color_discrete_sequence = salon_brand_color)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)


        if aggregate == 'Month':
            title = f"Monthly Services Provided in {selected_year}"
            fig = px.histogram(services_filtered, x='Bill_Month', title=title, color_discrete_sequence = salon_brand_color)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    

        st.markdown(
            "Most popular services this year were **{}** as it was taken **{}** times respectively each bringing a revenue of **{}**."
            .format(top_services, top_services_freq, top_services_rev 
         )
        )
    
    with chart3:

        products = tickets_details_df[tickets_details_df['Type'] == 'P']
        products['Bill_Date'] = pd.to_datetime(products['Bill_Date'])
        products['Bill_Month'] = products['Bill_Date'].dt.strftime('%B')

        products_filtered = products[products['Bill_Date'].dt.year == selected_year]

        products_counts = products_filtered.groupby('Descr').agg(
        frequency=('Qty', 'sum'),  
        total_sum=('Total', 'sum')).reset_index()
        products_counts = products_counts.sort_values(by='frequency', ascending=False)
        top_products_df = products_counts.head(3)
        top_products = list(top_products_df['Descr'])
        top_products = ', '.join("'" + element + "'" for element in top_products)
        top_products_freq = list(top_products_df['frequency']) 

        top_products_rev = list(top_products_df['total_sum']) 
        top_products_rev  = [f"₹{number:,.2f}" for number in top_products_rev]
        top_products_rev  = ', '.join(top_products_rev)

        if aggregate == 'Day':
            title = f"Daily Products Sold in {selected_year}"
            fig = px.histogram(products_filtered, x='Bill_Date', title=title, color_discrete_sequence = salon_brand_color)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)


        if aggregate == 'Month':
            title = f"Monthly Products Sold in {selected_year}"
            fig = px.histogram(products_filtered, x='Bill_Month', title=title, color_discrete_sequence = salon_brand_color)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        st.markdown(
            "Most popular products this year were **{}** that were purchased at a quantity of **{}** respectively each bringing a total revenue of **{}**"
            .format(top_products, top_products_freq, top_products_rev
        )
        )

    
    ## CHARTS ABOUT CLIENTS ## 

    cuscol1, cuscol2 = st.columns(2)

    with cuscol1:
        sales_df_filtered = sales_df[sales_df['Sex'].notna()]
        grouped = sales_df_filtered.groupby('Sex')['TotalSpend'].mean()

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
                    title=f"AVB by Gender in {selected_year}", color_discrete_sequence = salon_brand_color)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    with cuscol2:
        clients_df['Create_Date'] = pd.to_datetime(clients_df['Create_Date'])
        clients_filt = clients_df[clients_df['Create_Date'].dt.year == selected_year]

        clients = pd.DataFrame(clients_filt["category"].dropna().value_counts()).reset_index()
        clients.columns = ["Category", "Count"]

        title = f"Customer Types in {selected_year}"
        fig = px.pie(clients, values='Count', names='Category', title=title, color_discrete_sequence = salon_brand_color)

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    


def rfm_model():
    st.subheader("RFM Model for Customer Segmentation")
    st.write("The RFM (Recency, Frequency, Monetary) model is a powerful tool used in marketing to analyse customer behaviour and segment customers based on their purchasing patterns. Here’s a detailed explanation of how the RFM model works and how the tiers and buckets are categorised.")
    st.write("Recency (R) measures how recently a customer has made a purchase. Customers who have purchased more recently are considered more likely to purchase again. Frequency (F) Measures how often a customer makes a purchase. Customers who purchase more frequently are considered more loyal. Monetary (M): Measures how much money a customer spends on purchases. Customers who spend more money are considered more valuable.")
    
    tickets_details_df = st.session_state.tickets_details_df.copy()
    tickets_df = st.session_state.tickets_df.copy()
    clients_df = st.session_state.clients_df.copy()
    sales_df = st.session_state.sales_df.copy()

    sales_df["Monetary"] = sales_df["TotalSpend"]
    sales_df["Frequency"] = sales_df["NumVisits"]
    sales_df["Recency"] = sales_df["DaysSinceLastVisit"]


    ## NORMALIZE RFM VARIABLES ##
    customers_fix = pd.DataFrame()
    customers_fix["Recency"] = pd.Series(np.cbrt(sales_df['Recency'])).values
    customers_fix["Frequency"] = pd.Series(np.cbrt(sales_df['Frequency'])).values
    customers_fix["MonetaryValue"] = pd.Series(np.cbrt(sales_df['Monetary'])).values

    scaler = StandardScaler()
    scaler.fit(customers_fix)
    customers_normalized = scaler.transform(customers_fix)

    ## PER ELBOW METHOD ANALYSIS 4 SEEMS THE OPTIMAL NUMBER OF CLUSTERS ##
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(customers_normalized)

    ## LABEL CLIENTS ##
    sales_df.reset_index()  
    sales_df["Cluster"] = model.labels_

    df_normalized = pd.DataFrame(customers_normalized, columns=['Recency', 'Frequency', 'MonetaryValue'])
    df_normalized['ID'] = sales_df.index
    df_normalized['Cluster'] = sales_df['Cluster']
    df_normalized.head()

    df_nor_melt = pd.melt(df_normalized.reset_index(),
                      id_vars=['ID', 'Cluster'],
                      value_vars=['Recency','Frequency','MonetaryValue'],
                      var_name='Attribute',
                      value_name='Value')

    ## GATHER INFO ON CLUSTERS ##
    clusters_info = pd.DataFrame()
    clusters_info = df_nor_melt.groupby('Cluster').agg({'ID': lambda x: len(x)}).reset_index()
    clusters_info.rename(columns={'ID': 'Count'}, inplace=True)
    clusters_info['Percent'] = (clusters_info['Count'] / clusters_info['Count'].sum()) * 100
    clusters_info['Percent'] = clusters_info['Percent'].round(1)
    clusters_info['Percent'] = clusters_info['Percent'].map("{:.1f}%".format)


    sales_df.loc[:, "Cluster Segment"] = ""
    sales_df.loc[sales_df.loc[:, "Cluster"] == 3, "Cluster Segment"] = "New Customers"
    sales_df.loc[sales_df.loc[:, "Cluster"] == 2, "Cluster Segment"] = "At Risk Customers"
    sales_df.loc[sales_df.loc[:, "Cluster"] == 1, "Cluster Segment"] = "Loyal Customers"
    sales_df.loc[sales_df.loc[:, "Cluster"] == 0, "Cluster Segment"] = "Churned Customers"
    sales_df.reset_index(drop=True, inplace=True)

    clusters_stats = sales_df.groupby(['Cluster Segment', 'Cluster']).agg({
    'Recency':'mean',
    'Frequency':'mean',
    'Monetary':'mean'}).round(1).reset_index()

    clusters_about = pd.merge(clusters_info, clusters_stats, on='Cluster', how='inner')
    clusters_about = clusters_about[['Cluster', 'Cluster Segment', 'Count', 'Percent', 'Recency', 'Frequency', 'Monetary']]
    clusters_about.rename(columns={'Recency': 'Avg_Days_Since_Last_Visit', 'Frequency': 'Avg_Num_Visits', 'Monetary': 'Avg_Bill_Value'}, inplace=True)

    cluster_avg = sales_df[['Recency', 'Frequency', 'Monetary', 'Cluster Segment']].groupby('Cluster Segment').mean()

    population_avg = sales_df[['Recency', 'Frequency', 'Monetary']].mean()
    relative_imp = cluster_avg / population_avg - 1
    relative_imp.reset_index(inplace=True)
    relative_imp.rename(columns={'Recency': 'Relative Recency', 'Frequency': 'Relative Frequency', 'Monetary': 'Relative Monetary'}, inplace=True)
    relative_imp['Relative Recency'] = relative_imp['Relative Recency'].map("{:.0%}".format)
    relative_imp['Relative Frequency'] = relative_imp['Relative Frequency'].map("{:.0%}".format)
    relative_imp['Relative Monetary'] = relative_imp['Relative Monetary'].map("{:.0%}".format)

    clusters_dtl = pd.merge(clusters_about, relative_imp, on='Cluster Segment', how='inner')

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


    cluster1, cluster2 = st.columns([1.5, 1])

    with cluster1:
        st.write("To effectively segment customers, we use the K-Means clustering algorithm to group customers based on their normalised RFM scores.")
        st.write("""
        ### Customer Segments
        - **New Customers**: Recently acquired customers.
        - **At Risk Customers**: Customers who are not making frequent purchases.
        - **Loyal Customers**: Customers who purchase frequently and recently.
        - **Churned Customers**: Customers who have not made recent purchases.
        """)
        fig3 = sales_df.groupby('Cluster Segment').agg({'ClientID': lambda x: len(x)}).reset_index()
        fig3.rename(columns={'ClientID': 'Count'}, inplace=True)
        fig3['percent'] = (fig3['Count'] / fig3['Count'].sum()) * 100
        fig3['percent'] = fig3['percent'].round(1)
        colors=['#e3bcdf','#b84d9a','#bf7aac', '#8c0a7f'] #color palette

        fig = px.treemap(fig3, path=['Cluster Segment'],values='Count'
                        , width=800, height=700
                        ,title="AI Generated Clusters from RFM Values")

        fig.update_layout(
            treemapcolorway = colors, #defines the colors in the treemap
            margin = dict(t=50, l=25, r=25, b=25))

        fig.data[0].textinfo = 'label+text+value+percent root'
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    with cluster2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        clusters_dtl.drop(columns=['Cluster'], inplace=True)
        st.dataframe(clusters_dtl)

        sns.set_style("whitegrid", {'grid.color': '.95'})
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.lineplot(x='Attribute', y='Value', hue='Cluster Segment', data=df_nor_melt, ax=ax)
        st.pyplot(fig)
    
    rfmseg1, rfmseg2 = st.columns([1.5, 1])

    with rfmseg1:
        st.write('### RFM Segments')
        st.write("To further categorise customers, we calculate an RFM score for each customer and group them into the following categories.")

        # Aggregate data by each customer
        fig4 = sales_df.groupby('Segment').agg({'ClientID': lambda x: len(x)}).reset_index()

        # Rename columns
        fig4.rename(columns={'ClientID': 'Count'}, inplace=True)
        fig4['percent'] = (fig4['Count'] / fig4['Count'].sum()) * 100
        fig4['percent'] = fig4['percent'].round(1)


        colors=['#713ebd','#9771d1','#7d5cad','#a386cf','#6f1aed','#7b38e0','#c2aae6','#6b0ec2'] #color palette

        fig = px.treemap(fig4, path=['Segment'],values='Count'
                        , width=800, height=500
                        )

        fig.update_layout(
            treemapcolorway = colors, #defines the colors in the treemap
            margin = dict(t=50, l=25, r=25, b=25))

        fig.data[0].textinfo = 'label+text+value+percent root'
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


    with rfmseg2:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")

        segment_descriptions = {
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


        rfmseg_df = sales_df.groupby(['Segment']).agg({'ClientID': lambda x: len(x), 'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean' }).reset_index()
        rfmseg_df.rename(columns={'ClientID': 'Customer_Count', 'Recency': 'Avg_Days_Since_Last_Visit', 'Frequency': 'Avg_Visits', 'Monetary': 'Avg_Bill_Value'}, inplace=True)
        rfmseg_df['Description'] = rfmseg_df['Segment'].map(segment_descriptions)

        st.write(rfmseg_df)


    col1, col2 = st.columns([1.5, 1])

    clients_subset = clients_df[['ClientID', 'FirstName', 'LastName', 'HomePhone', 'Sex', 'DOB', 'survey', 'MembershipCardNo',
                        'Membership_Date']]
    with col1:
        st.write("### Customer Loyalty Tiers")
        st.write("Based on the RFM score, we assign customers to different loyalty tiers.")

        fig5 = sales_df.groupby('Score').agg({'ClientID': lambda x: len(x)}).reset_index()

        # Rename columns
        fig5.rename(columns={'ClientID': 'Count'}, inplace=True)
        fig5['percent'] = (fig5['Count'] / fig5['Count'].sum()) * 100
        fig5['percent'] = fig5['percent'].round(1)

        colors=['#613787','#9d81b8','#7717d1','#7d6296', '#8400ff'] #color palette

        fig = px.treemap(fig5, path=['Score'],values='Count'
                        , width=800, height=400
                    )

        fig.update_layout(
            treemapcolorway = colors, #defines the colors in the treemap
            margin = dict(t=50, l=25, r=25, b=25))

        fig.data[0].textinfo = 'label+text+value+percent root'
        st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    
    with col2:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")

        tier_descriptions = {
            'Platinum': 'Top tier customers with the highest RFM scores, representing the most valuable and loyal customers.',
            'Gold': 'High tier customers with strong loyalty and value, just below Platinum.',
            'Silver': 'Mid-tier customers with moderate loyalty and value.',
            'Bronze': 'Lower tier customers who are less engaged or at risk of churning.',
            'Green': 'New or low-engagement customers who haven’t yet shown strong purchasing behavior.'
        }

        loyalty_df = sales_df.groupby(['Score']).agg({'ClientID': lambda x: len(x), 'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean' }).reset_index()
        loyalty_df.rename(columns={'ClientID': 'Customer_Count', 'Recency': 'Avg_Days_Since_Last_Visit', 'Frequency': 'Avg_Visits', 'Monetary': 'Avg_Bill_Value'}, inplace=True)
        loyalty_df.rename(columns={'Score': 'Tier'}, inplace=True)
        loyalty_df['description'] = loyalty_df['Tier'].map(tier_descriptions)

        st.dataframe(loyalty_df, width=1200)

    st.write("### Download Client Data by Loyalty Tier")

    best1, best2 = st.columns(2)

    with best1:
        st.markdown("**Tier 1: Platinum Customers**")
        platinum_customers = sales_df[sales_df['Score'] == "Platinum"]
        platinum_customers = platinum_customers.rename(columns={'Recency': 'Days since last visit', 'Frequency': 'Number of Visits',
                                'Monetary': 'Total Billed'})
        platinum_customers['AVB'] = platinum_customers.groupby(['ClientID'])['Total Billed'].transform('mean')
        st.dataframe(platinum_customers.sample(3))
        csv = convert_df(platinum_customers)
        st.download_button(
            label="Download platinum customer data",
            data=csv,
            file_name='platinum_customers.csv',
            mime='text/csv', use_container_width=True
        )
    
    with best2:
        
        st.markdown("**Tier 2: Gold Customers**")
        gold_customers = sales_df[sales_df['Score'] == "Gold"]
        gold_customers = gold_customers.rename(columns={'Recency': 'Days since last visit', 'Frequency': 'Number of Visits',
                        'Monetary': 'Total Billed'})
        gold_customers['AVB'] = gold_customers.groupby(['ClientID'])['Total Billed'].transform('mean')
        st.dataframe(gold_customers.sample(3))
        csv = convert_df(gold_customers)
        st.download_button(
            label="Download gold customer data",
            data=csv,
            file_name='gold_customers.csv',
            mime='text/csv', use_container_width=True
        )

    secondb1, secondb2 = st.columns(2)

    with secondb1:
        st.markdown("**Tier 3: Silver Customers**")
        silver_customers = sales_df[sales_df['Score'] == "Silver"]
        silver_customers = silver_customers.rename(columns={'Recency': 'Days since last visit', 'Frequency': 'Number of Visits',
                        'Monetary': 'Total Billed'})
        silver_customers['AVB'] = silver_customers.groupby(['ClientID'])['Total Billed'].transform('mean')
        st.dataframe(silver_customers.sample(3))
        csv = convert_df(silver_customers)
        st.download_button(
            label="Download silver customer data",
            data=csv,
            file_name='silver_customers.csv',
            mime='text/csv', use_container_width=True
        )

    with secondb2:
        st.markdown("**Tier 4: Bronze Customers**")
        bronze_customers = sales_df[sales_df['Score'] == "Bronze"]
        bronze_customers = bronze_customers.rename(columns={'Recency': 'Days since last visit', 'Frequency': 'Number of Visits',
                'Monetary': 'Total Billed'})
        bronze_customers['AVB'] = bronze_customers.groupby(['ClientID'])['Total Billed'].transform('mean')
        st.dataframe(bronze_customers.sample(3))
        csv = convert_df(bronze_customers)
        st.download_button(
            label="Download bronze customer data",
            data=csv,
            file_name='bronze_customers.csv',
            mime='text/csv', use_container_width=True
        )


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_data
def load_and_process_data():
    logger.info("Loading and processing data")
    tickets_df = pd.read_csv("data/Tickets_14Nov23_4pm.csv", encoding='ISO-8859-1', low_memory=False)
    tickets_details_df = pd.read_excel("data/New_Ticket_Product_Details_14Nov_23.xlsx")

    tickets_df['Bill_Date'] = pd.to_datetime(tickets_df['Created_Date'])
    tickets_details_df['Bill_Date'] = pd.to_datetime(tickets_details_df['Created_Date2'])

    tickets_details_df['TicketID'] = tickets_details_df['TicketID'].astype('int64', errors='ignore')
    tickets_df = tickets_df[(tickets_df['TicketID'] != 'Closed') & (tickets_df['TicketID'].notna())]
    tickets_df['TicketID'] = tickets_df['TicketID'].astype('int64', errors='ignore')

    return tickets_df, tickets_details_df

@st.cache_data
def filter_and_merge_data(tickets_df, tickets_details_df):
    tickets_details_df = tickets_details_df[~tickets_details_df['Group1'].isna()]
    client_services = pd.merge(tickets_details_df, tickets_df, on='TicketID', how='left')
    client_services['Frequency'] = client_services.groupby(['ClientID', 'Descr'])['ClientID'].transform('size')
    client_services.reset_index(inplace=True)

    return client_services

@st.cache_data
def create_pivot_table(client_services):
    client_services = client_services[['ClientID', 'Descr', 'Frequency']]
    all_pivots = client_services.pivot_table(index='ClientID', columns='Descr', values='Frequency', fill_value=0)
    pivot_df_sample = all_pivots.sample(100, random_state=42)
    return pivot_df_sample

def recommendation_model():
    logger.info("Starting recommendation model")
    st.subheader("Recommendation Model")
    st.markdown("Content-Based recommendation model based on cosine similarity of services taken by clients.")

    # Load and process data
    start_time = datetime.now()
    tickets_df, tickets_details_df = load_and_process_data()
    logger.info(f"Data loaded and processed in {datetime.now() - start_time}")

    tickets_details_df['TicketID'] = tickets_details_df['TicketID'].astype('int64', errors='ignore')
    tickets_df['TicketID'] = tickets_df['TicketID'].astype('int64', errors='ignore')

    client_services = filter_and_merge_data(tickets_df, tickets_details_df)
    pivot_df_sample = create_pivot_table(client_services)
    logger.info(f"Pivot table created and sampled in {datetime.now() - start_time}")

    col1, col2 = st.columns(2)

    # Select confidence threshold first
    threshold = st.select_slider(
        '**Select a Confidence Score Threshold**',
        options=[0.25, 0.50, 0.75, 0.95],
        value=0.5
    )
    st.write(
        "The confidence score sets how similar recommendations need to be: lower scores give more options, higher scores give more accurate ones.")

    if threshold:
        logger.info("Threshold selected, generating recommendations.")

        # Calculate cosine similarity
        start_time = datetime.now()
        customer_similarity = cosine_similarity(pivot_df_sample)
        logger.info(f"Cosine similarity calculated in {datetime.now() - start_time}")

        # Generate recommendations
        recommendations = {}
        for idx, customer in enumerate(pivot_df_sample.index):
            similar_customers = sorted(list(enumerate(customer_similarity[idx])), key=lambda x: x[1], reverse=True)
            recommendations[customer] = []
            for similar_customer, similarity_score in similar_customers[1:]:  # Exclude the customer itself
                if similarity_score > threshold:  # Check if similarity score is above the threshold
                    similar_products = pivot_df_sample.columns[pivot_df_sample.iloc[similar_customer] > 0].tolist()
                    for product in similar_products:
                        if pivot_df_sample.loc[
                            customer, product] == 0:  # Check if the customer hasn't bought the product yet
                            recommendations[customer].append(
                                (product, similarity_score))  # Store product and similarity score

        # Filter customers with recommendations
        cust_list_with_recommendations = [
            customer for customer, recs in recommendations.items() if len(recs) > 0
        ]

        # Show customer select box only when filtered customers are available
        selected_customer_name = st.selectbox(
            "Select a Customer with Recommendations",
            cust_list_with_recommendations
        )

        # Display recommendations for the selected customer
        if selected_customer_name and selected_customer_name in recommendations and recommendations[
            selected_customer_name]:
            st.divider()

            # Get the services the customer has taken (maximum 3)
            customer_services = pivot_df_sample.columns[pivot_df_sample.loc[selected_customer_name] > 0].tolist()[
                                :3]
            services_taken = ", ".join(customer_services)

            # Get the top 5 recommendations
            top_recommendations = recommendations[selected_customer_name][:5]
            recommendations_list = [f"{index + 1}. {item[0]} (Confidence: {item[1]:.2f})"
                                    for index, item in enumerate(top_recommendations)]

            # Display the personalized message
            st.write(f"**Dear {selected_customer_name},**")
            st.write(f"Because you've taken the following services: {services_taken}, we think you'll enjoy these:")
            for rec in recommendations_list:
                st.write(rec)

            # Display confidence report in a dataframe
            df = pd.DataFrame(top_recommendations, columns=["Service Description", "Confidence Score"])
            st.write("**Confidence Report**")
            st.dataframe(df, width=1000, hide_index=True)
        else:
            st.write("Sorry, no strong recommendations are available for this customer.")

    # Add download button for pivot_df_sample
    csv = pivot_df_sample.to_csv(index=True).encode('utf-8')
    st.write(" ")
    st.write("------------------")
    col11, col12 = st.columns([85, 15])
    col12.download_button(
        label="Download Data",
        data=csv,
        file_name='pivot_table.csv',
        mime='text/csv',
    )
    col11.write("Download the pivot table containing the cosine similarity scores for all customers against all services:")




page_names_to_funcs = {
    "—": intro,
    "Business Dashboard Demo": business_dashboard,
    "RFM Model Demo": rfm_model,
    "Recommendation Model Demo": recommendation_model
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

#if __name__ == "__main__":
   # main()