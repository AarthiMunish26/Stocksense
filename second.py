import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Load SBERT model (cached for efficiency)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Function to fetch stock price from Google Finance
def get_stock_price(ticker):
    try:
        url = f"https://www.google.com/finance/quote/{ticker}:NSE"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return "Error fetching data. Try again later."

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract stock price
        price_tag = soup.find("div", class_="YMlKec fxKbKc")
        if price_tag:
            return price_tag.text
        else:
            return "Stock not found or data unavailable."

    except Exception as e:
        return f"Error: {str(e)}"

# --- Streamlit App ---
st.title("StockSense: Stock Market Analysis & Investment Recommendations")

# Sidebar for navigation and user input
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Trend Analysis",
    "Anomaly Detection",
    "Profit/Loss Analysis",
    "Location-Based Visualization",
    "Search Relevancy & Investment Recommendations",
    "Live Stock Price"  # New page for live stock price
])

# Sidebar: Upload Dataset
st.sidebar.header("Upload CSV Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a stock market dataset (CSV)", type=["csv"])

# Load CSV if uploaded
if uploaded_file:
    try:
        stock_data = pd.read_csv(uploaded_file)
        required_columns = ['Date', 'Close', 'Search_Relevancy_Query', 'Sector', 'Company', 'Location']
        if all(col in stock_data.columns for col in required_columns):
            stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
            stock_data = stock_data.dropna(subset=['Date']).sort_values(by='Date')
        else:
            st.error("⚠️ CSV must contain 'Date', 'Close', 'Search_Relevancy_Query', 'Sector', 'Company', and 'Location' columns!")
            stock_data = None
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        stock_data = None
else:
    stock_data = None

# --- User Input Features ---
if stock_data is not None:
    st.sidebar.header("User Input Features")

    # Sector filter
    selected_sector = st.sidebar.selectbox("Select Sector", ["All"] + list(stock_data['Sector'].unique()))
    if selected_sector == "All":
        sector_data = stock_data
    else:
        sector_data = stock_data[stock_data['Sector'] == selected_sector]

    # Company filter
    selected_company = st.sidebar.selectbox("Select Company", ["All"] + list(sector_data['Company'].unique()))
    if selected_company == "All":
        company_data = sector_data
    else:
        company_data = sector_data[sector_data['Company'] == selected_company]

    # Date range filter
    st.sidebar.header("Filter by Date Range")
    min_date = stock_data['Date'].min()
    max_date = stock_data['Date'].max()
    start_date, end_date = st.sidebar.date_input("Select date range", [min_date, max_date])

    # Filter data based on user input
    filtered_data = company_data[(company_data['Date'] >= pd.to_datetime(start_date)) & 
                                 (company_data['Date'] <= pd.to_datetime(end_date))]
else:
    filtered_data = None

# --- Home Page ---
if page == "Overview":
    st.header("Welcome to the StockSense")
    st.write("Start analyzing trends and search relevancy.")
    st.write("Combines Stock and Sense, implying the app helps users make sense of stock market data.")

# --- Trend Analysis Page ---
elif page == "Trend Analysis":
    if filtered_data is not None:
        st.header("Trend Analysis")

        # 1. Line chart for Close prices over time
        st.subheader("Close Price Over Time")
        fig_close_price = px.line(filtered_data, x='Date', y='Close', title='Close Price Over Time', 
                                  labels={'Date': 'Date', 'Close': 'Close Price'}, markers=True)
        st.plotly_chart(fig_close_price)

        # 2. Bar chart for Volume traded over time
        st.subheader("Volume Traded Over Time")
        fig_volume = px.bar(filtered_data, x='Date', y='Volume', title='Volume Traded Over Time', 
                            labels={'Date': 'Date', 'Volume': 'Volume Traded'})
        st.plotly_chart(fig_volume)

        # 3. Distribution of Close prices
        st.subheader("Distribution of Close Prices")
        fig_distribution = px.histogram(filtered_data, x='Close', nbins=30, title="Close Price Distribution", 
                                        labels={'Close': 'Close Price'})
        st.plotly_chart(fig_distribution)

        # 4. Moving Average for Close price
        st.subheader("Moving Average (7-Day) for Close Price")
        filtered_data['7-Day Moving Average'] = filtered_data['Close'].rolling(window=7).mean()
        fig_moving_avg = px.line(filtered_data, x='Date', y=['Close', '7-Day Moving Average'], 
                                 title='Close Price with 7-Day Moving Average', 
                                 labels={'Date': 'Date', 'value': 'Price'}, 
                                 line_dash_sequence=['solid', 'dash'])
        st.plotly_chart(fig_moving_avg)

        # 5. Enhanced Candlestick chart for stock prices
        st.subheader("Enhanced Candlestick Chart with Volume")
        fig_candlestick = go.Figure(data=[go.Candlestick(
            x=filtered_data['Date'],
            open=filtered_data['Open'],
            high=filtered_data['High'],
            low=filtered_data['Low'],
            close=filtered_data['Close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        fig_candlestick.add_trace(go.Bar(x=filtered_data['Date'], y=filtered_data['Volume'], 
                                         name='Volume', opacity=0.3))
        fig_candlestick.update_layout(title="Candlestick Chart with Volume", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_candlestick)

        # 6. Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_data = filtered_data.select_dtypes(include=[np.number])
        corr = numeric_data.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', title='Correlation Heatmap')
        st.plotly_chart(fig_corr)
    else:
        st.warning("⚠️ Please upload a valid dataset to view trend analysis.")

# --- Profit/Loss Analysis Page ---
elif page == "Profit/Loss Analysis":
    if filtered_data is not None:
        st.title("Profit/Loss Analysis")

        # Calculate profit/loss based on previous close price
        filtered_data['Prev_Close'] = filtered_data['Close'].shift(1)
        filtered_data['Profit/Loss'] = filtered_data['Close'] - filtered_data['Prev_Close']
        filtered_data['Status'] = np.where(filtered_data['Profit/Loss'] > 0, 'Profit', 'Loss')

        # Filter by profit or loss
        profit_loss_filter = st.sidebar.radio("Filter by Profit/Loss", ["All", "Profit", "Loss"])
        if profit_loss_filter != "All":
            filtered_data = filtered_data[filtered_data['Status'] == profit_loss_filter]

        st.write("### Profit/Loss Data")
        st.write(filtered_data)

        # Visualize profit/loss over time
        fig_profit_loss = px.bar(filtered_data, x='Date', y='Profit/Loss', color='Status',
                                 title="Profit/Loss Over Time", 
                                 labels={'Date': 'Date', 'Profit/Loss': 'Profit/Loss Amount'})
        st.plotly_chart(fig_profit_loss)
    else:
        st.warning("⚠️ Please upload a valid dataset to perform profit/loss analysis.")

# --- Location-Based Visualization Page ---
elif page == "Location-Based Visualization":
    if filtered_data is not None:
        st.title("Location-Based Visualization")

        # Ensure the data has a 'Location' column
        if 'Location' in filtered_data.columns:
            location_filter = st.sidebar.selectbox("Select Location", ["All"] + list(filtered_data['Location'].unique()))
            if location_filter == "All":
                location_data = filtered_data
            else:
                location_data = filtered_data[filtered_data['Location'] == location_filter]

            st.write(f"### Companies in {location_filter}")
            st.write(location_data)

            # Visualize average close price by company in the selected location
            avg_close_by_company = location_data.groupby('Company')['Close'].mean().reset_index()
            fig_avg_close_location = px.bar(avg_close_by_company, x='Company', y='Close', 
                                            title=f"Average Close Price by Company in {location_filter}", 
                                            labels={'Company': 'Company', 'Close': 'Average Close Price'})
            st.plotly_chart(fig_avg_close_location)
        else:
            st.write("Location data is not available in the dataset.")
    else:
        st.warning("⚠️ Please upload a valid dataset to perform location-based visualization.")

# --- Search Relevancy & Investment Decision Page ---
elif page == "Search Relevancy & Investment Recommendations":
    if filtered_data is not None:
        st.header("🔎 Search Relevancy & Investment Decision System")
        if filtered_data['Search_Relevancy_Query'].isnull().any():
            st.warning("⚠️ Some queries are missing. These rows will not be included in the relevancy analysis.")
            filtered_data.dropna(subset=['Search_Relevancy_Query'], inplace=True)

        filtered_data['embedding'] = filtered_data['Search_Relevancy_Query'].apply(lambda x: model.encode(str(x), convert_to_tensor=True))
        search_query = st.text_input("Enter a search query (e.g., technology, cloud, AI)")

        if st.button("Search & Predict Investment Decision"):
            if search_query:
                query_embedding = model.encode(search_query, convert_to_tensor=True)
                filtered_data['similarity'] = filtered_data['embedding'].apply(lambda x: util.pytorch_cos_sim(query_embedding, x).item())
                search_results = filtered_data.sort_values(by='similarity', ascending=False)
                best_match = search_results.iloc[0]
                
                st.subheader("🏆 Best Match")
                st.markdown(f"**🏢 Company:** {best_match['Company']}")
                st.markdown(f"**📍 Sector:** {best_match['Sector']}")
                st.markdown(f"**📍 Location:** {best_match['Location']}")
                st.markdown(f"**🔍 Query Match:** {best_match['Search_Relevancy_Query']}")
                st.markdown(f"**🔗 Best Relevancy Score:** {best_match['similarity']:.4f}")

                # Investment Decision Logic
                threshold_buy = 0.75
                threshold_sell = 0.4

                def investment_decision(score):
                    if score >= threshold_buy:
                        return "BUY"
                    elif score <= threshold_sell:
                        return "SELL"
                    else:
                        return "HOLD"

                search_results['Decision'] = search_results['similarity'].apply(investment_decision)
                st.write("💡 **Investment Recommendations**")
                st.dataframe(search_results[['Company', 'Sector', 'Location', 'Search_Relevancy_Query', 'similarity', 'Decision']].head(10))
                
                # Visualize Stock Price Trends for Top Matches
                st.subheader(f"📊 Stock Price Trend for '{search_query}'")
                top_matches = search_results.head(10)
                fig_filtered = px.line(top_matches, x='Date', y='Close', title=f"Stock Price Trend for '{search_query}'",
                                       labels={'Close': 'Price (USD)', 'Date': 'Date'},
                                       hover_data={'Company': True, 'Sector': True, 'Location': True})
                fig_filtered.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_filtered, use_container_width=True)
            else:
                st.warning("Please enter a search query to perform relevance analysis.")
    else:
        st.warning("⚠️ Please upload a valid dataset to use the search relevancy system.")

# --- Anomaly Detection Page ---
elif page == "Anomaly Detection":
    if filtered_data is not None:
        st.title("Anomaly Detection")
        anomaly_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        anomaly_data = filtered_data[anomaly_features].dropna()

        # Train Isolation Forest model
        isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        anomaly_data['Anomaly'] = isolation_forest.fit_predict(anomaly_data)
        anomaly_data['Anomaly'] = anomaly_data['Anomaly'].map({-1: 'Anomaly', 1: 'Normal'})

        # Merge anomaly data with filtered data
        filtered_data = filtered_data.reset_index(drop=True)
        anomaly_data = pd.concat([filtered_data, anomaly_data['Anomaly']], axis=1)

        # Display anomaly results
        st.write("### Anomaly Detection Results")
        st.write(anomaly_data)

        # Visualize anomalies on Close Price over time
        anomaly_fig = px.scatter(anomaly_data, x='Date', y='Close', color='Anomaly', 
                                 title='Anomaly Detection in Close Price Over Time', 
                                 labels={'Date': 'Date', 'Close': 'Close Price'})
        st.plotly_chart(anomaly_fig)
    else:
        st.warning("⚠️ Please upload a valid dataset to perform anomaly detection.")

# --- Live Stock Price Page ---
elif page == "Live Stock Price":
    st.title("📈 Live Stock Price")

    # Display current date and time
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.write(f"🕒 Current Date & Time: {date_time}")

    # Predefined list of NSE stock tickers
    nse_stocks = [
    "NESTLEIND", "TCS", "NTPC", "TATAPOWER", "ITC", "SBIN", "SUNPHARMA", "ICICIBANK", "ONGC",
    "APOLLOHOSP", "AXISBANK", "BRITANNIA", "TECHM", "ADANIGREEN", "KOTAKBANK", "HCLTECH",
    "CIPLA", "DRREDDY", "BIOCON", "WIPRO", "HDFCBANK", "HINDUNILVR", "INFY", "MARICO", "RELIANCE"
    ]

    # Dropdown select box for stock ticker
    ticker = st.selectbox("Select an NSE Stock Ticker:", nse_stocks)

    if st.button("Get Stock Price"):
        stock_price = get_stock_price(ticker)
        st.subheader(f"Stock Price for {ticker}: {stock_price}")
        st.write(f"📅 Retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")