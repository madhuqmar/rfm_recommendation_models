from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationModel:
    def __init__(self, sales_df, user_col, date_col, spend_col):
        self.sales_df = sales_df
        self.user_col = user_col
        self.date_col = date_col
        self.spend_col = spend_col

    def transform_data(self):
        pass

    def filter_and_merge_data(self, tickets_df, tickets_details_df):
        pass

    def create_pivot_table(self, client_services):
        pass

    def recommendation_model(self):

        # Load and process data
        start_time = datetime.now()
        tickets_df, tickets_details_df = load_and_process_data()

        tickets_details_df['TicketID'] = tickets_details_df['TicketID'].astype('int64', errors='ignore')
        tickets_df['TicketID'] = tickets_df['TicketID'].astype('int64', errors='ignore')

        client_services = self.filter_and_merge_data(tickets_df, tickets_details_df)
        pivot_df_sample = self.create_pivot_table(client_services)
        logger.info(f"Pivot table created and sampled in {datetime.now() - start_time}")

        # Select confidence threshold first
        threshold = st.select_slider(
            '**Select a Service Match Score (Threshold)- We Recommend setting it at 0.75**',
            options=[0.25, 0.50, 0.75, 0.95],
            value=0.75
        )

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
                "Select a Customer",
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
                recommendations_list = [f"{index + 1}. {item[0]} - {item[1] * 100:.0f}% Match"
                                        for index, item in enumerate(top_recommendations)]

                # Display the personalized message
                st.write(f"**Dear {selected_customer_name},**")
                st.write(f"Because you've taken: {services_taken} in the past, we recommend the following services based on what similar customers like you have taken and enjoyed — it could be a great addition to your beauty routine!:")
                for rec in recommendations_list:
                    st.write(rec)

                # Display confidence report in a dataframe
                df = pd.DataFrame(top_recommendations, columns=["Service Description", "Confidence Score"])
                st.write("**Confidence Report**")

                df = df.rename(columns={"Confidence Score": "Confidence Match"})
                df["Confidence Match"] = (df["Confidence Match"] * 100).round(0).astype(int).astype(str) + "%"
                st.dataframe(df, width=1000, hide_index=True)
            else:
                st.write("Sorry, no strong recommendations are available for this customer.")