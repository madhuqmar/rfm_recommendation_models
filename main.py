import streamlit as st
import pandas as pd 

st.set_page_config(layout="wide")

def main():
    st.title("BGorgeous")
    st.subheader("Customer Behavior Analysis", divider='violet')

    tickets_df = pd.read_csv("data/Tickets_14Nov23_4pm.csv", encoding='ISO-8859-1', low_memory=False)
    #st.dataframe(tickets_df)

    col1, col2, col3 = st.columns(3)

    tickets_df['Total'] = tickets_df['Total'].fillna(0)
    total_sales = sum(tickets_df['Total'])
    col1.metric("Total Sales", f"₹{total_sales:,.2f}")

    try:
        tickets_df['TicketID'] = tickets_df['TicketID'].astype('int64')
    except ValueError:
        tickets_df['TicketID'] = pd.to_numeric(tickets_df['TicketID'], errors='coerce')
    total_services = tickets_df['TicketID'].count()
    col2.metric("Total Services", f"{total_services:,}")

    #total_customers = 
    #col3.metric("Total Customers", )


if __name__ == "__main__":
    main()