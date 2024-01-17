import streamlit as st
import pandas as pd 
import plotly.express as px
import math
import numpy as np
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(layout="wide")

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def main():
    st.title("BGorgeous")
    st.subheader("Customer Behavior Analysis", divider='violet')
    st.markdown("Data Refresh Date: 14 November 2023")

    tickets_df = pd.read_csv("data/Tickets_14Nov23_4pm.csv", encoding='ISO-8859-1', low_memory=False)
    tickets_details_df = pd.read_csv("data/Ticket_Product_Details_14Nov23_4pm.csv", encoding='ISO-8859-1', 
            low_memory=False)

    clients_df = pd.read_csv("data/Client_14Nov23_4pm.csv", encoding='ISO-8859-1', low_memory=False)
    clients_df.dropna(subset=['ClientID'], inplace=True)
    exclude_values = ['.8220484146', '.8393863665', '.9894384197', 'C Balachander9884886817', '0', '..8220484146']
    clients_df = clients_df[~clients_df['ClientID'].isin(exclude_values)]

    col1, col2, col3, col4 = st.columns(4)

    tickets_df['Total'] = tickets_df['Total'].fillna(0)
    total_sales = sum(tickets_df['Total'])
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
    selected_year = st.selectbox('Select Year', year_list)

    tickets_details_df['Bill_DateTime'] = pd.to_datetime(tickets_details_df['Start_Time'], format='%d-%m-%Y %H:%M', errors='coerce')
    tickets_details_df['Bill_Date'] = tickets_details_df['Bill_DateTime'].dt.date
    tickets_details_df['Bill_Time'] = tickets_details_df['Bill_DateTime'].dt.time
    tickets_details_df = tickets_details_df.drop(columns=['Bill_DateTime', 'Start_Time'])

    chart1, chart2, chart3 = st.columns(3)

    with chart1:
        
        st.subheader(f"Total Sales in {selected_year}")
        tickets_filt = tickets_df[tickets_df['Bill_Date'].dt.year == selected_year]
        
        title = f"Total Sales Made in {selected_year}"
        fig = px.bar(
                tickets_filt, 
                x = 'Bill_Date', 
                y = 'Total', 
                title = title, 
                color_discrete_sequence = ["#8633de"])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    with chart2:
        st.subheader(f"Total Services in {selected_year}")
        services = tickets_details_df[tickets_details_df['Type'] == 'S']
        services['Bill_Date'] = pd.to_datetime(services['Bill_Date'], format='%d-%m-%Y')
        services_filtered = services[services['Bill_Date'].dt.year == selected_year]
        fig = px.histogram(services_filtered, x='Bill_Date', title='Services by Date', color_discrete_sequence = ["#8633de"])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    with chart3:
        st.subheader(f"Total Products Sold in {selected_year}")
        products = tickets_details_df[tickets_details_df['Type'] == 'P']
        products['Bill_Date'] = pd.to_datetime(products['Bill_Date'], format='%d-%m-%Y')
        products_filtered = products[products['Bill_Date'].dt.year == selected_year]
        fig = px.histogram(products_filtered, x='Bill_Date', title='Products sold by Date', color_discrete_sequence = ["#8633de"])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


    
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
        client_services_df['Bill_Date'] = pd.to_datetime(client_services_df['Bill_Date'], format='%d-%m-%Y')
        client_services_df_filtered = client_services_df[client_services_df['Bill_Date'].dt.year == selected_year]
        grouped = client_services_df_filtered.groupby('Sex')['Total'].median()
        gender_labels = {
        "F": 'Female',
        "M": 'Male'
    }
        df = pd.DataFrame({
            'Gender': [gender_labels[x] for x in grouped.index],
            'Median Spend': grouped.values
        })

        st.subheader(f"Median Spend by Gender in {selected_year}")
        fig = px.bar(df, x='Median Spend', y='Gender', orientation='h', 
                    labels={'Median Spend': 'Median Spend', 'Gender': 'Gender'},
                    title=f"Median Spend by Gender in {selected_year}", color_discrete_sequence = ["#8633de"])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    with cuscol2:
        clients_df['Create_Date'] = pd.to_datetime(clients_df['Create_Date'])
        clients_filt = clients_df[clients_df['Create_Date'].dt.year == selected_year]

        clients = pd.DataFrame(clients_filt["category"].dropna().value_counts()).reset_index()
        clients.columns = ["Category", "Count"]

        st.subheader(f"Types of Customers in {selected_year}")
        title = f"Customer Types in {selected_year}"
        fig = px.pie(clients, values='Count', names='Category', title=title, color_discrete_sequence = ["#8633de"])

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


    st.divider()

    st.subheader("RFM Model")

    cutoff_date = tickets_df['Bill_Date'].max()
    tickets_df['Num_Visits'] = tickets_df.groupby('ClientID')['ClientID'].transform('count')

    fm_df = tickets_df.groupby('ClientID').agg({'Num_Visits': 'sum', 'Total': 'sum'}).reset_index()
    fm_df["Frequency"] = tickets_df["Num_Visits"]
    fm_df["Monetary"] = tickets_df["Total"]
    fm_df = fm_df.drop_duplicates()

    tickets_df['Recency'] = (cutoff_date - tickets_df.groupby('ClientID')['Bill_Date'].transform('max')).dt.days
    recency_df = tickets_df[['ClientID', 'Recency']]
    recency_df = recency_df.drop_duplicates()
    
    rfm_df = pd.merge(fm_df, recency_df, on='ClientID', how='outer')
    rfm_df = pd.merge(clients_df, rfm_df, on='ClientID', how='left')

    rfm_df['RecencyScore'] = pd.qcut (rfm_df['Recency'], q = 5, labels = ['5', '4', '3', '2', '1'])
    rfm_df['FrequencyScore'] = pd.qcut (rfm_df['Frequency'], q = 5, labels = ['1', '2', '3', '4', '5'])
    rfm_df['MonetaryScore'] = pd.qcut (rfm_df['Monetary'], q = 5, labels = ['1', '2', '3', '4', '5'])

    rfm_df = rfm_df.dropna(subset=['RecencyScore', 'FrequencyScore', 'MonetaryScore'])

    
    rfm_df.loc[:, 'RFM_score'] = rfm_df['RecencyScore'].astype(int) + rfm_df['MonetaryScore'].astype(int) + rfm_df['FrequencyScore'].astype(int)

    segment_labels = ["Low-Value", "Mid-Value", "High-Value"]
    rfm_df.loc[:, "Value_Segment"] = pd.qcut(rfm_df["RFM_score"], q=3, labels=segment_labels)
    #st.dataframe(rfm_df[['ClientID', 'Value_Segment', 'RFM_score', 'Recency', 'Frequency', 'Monetary', 'RecencyScore', 'FrequencyScore', 'MonetaryScore']])

    value1, value2 = st.columns(2)

    with value1:

        st.subheader("Customer Value Segments")
        st.markdown("All time")
        value_segments_df = pd.DataFrame(rfm_df["Value_Segment"].value_counts()).reset_index()
        value_segments_df.columns = ["Value_Segment", "Count"]
        fig = px.bar(
            value_segments_df,
            x="Value_Segment",
            y="Count",
            title="Customers by Value Segment",
            color_discrete_sequence=["#8633de"],
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


    with value2:
        #col1, _, = st.columns(2)

       # with col1:
        st.subheader("High Value Customers")
        high_val_customers = rfm_df[rfm_df['Value_Segment'] == "High-Value"]
        st.dataframe(high_val_customers.sample(3))
        csv = convert_df(high_val_customers)
        st.download_button(
            label="Download high value customer data",
            data=csv,
            file_name='high_value_customers.csv',
            mime='text/csv', use_container_width=True
        )


        #with col1:
        st.subheader("Mid Value Customers")
        mid_val_customers = rfm_df[rfm_df['Value_Segment'] == "Mid-Value"]
        st.dataframe(mid_val_customers.sample(3))
        csv = convert_df(mid_val_customers)
        st.download_button(
            label="Download mid value customer data",
            data=csv,
            file_name='mid_value_customers.csv',
            mime='text/csv', use_container_width=True
        )

       # with col1:
           # st.subheader("Low Value Customers")
            #low_val_customers = rfm_df[rfm_df['Value_Segment'] == "Low-Value"]
            #st.dataframe(low_val_customers.sample(5))
            #csv = convert_df(low_val_customers)
           # st.download_button(
              #  data=csv,
              # file_name='low_value_customers.csv',
              #  mime='text/csv',
          #  )



if __name__ == "__main__":
    main()