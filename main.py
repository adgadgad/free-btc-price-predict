import json
import random
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import numpy as np
import pandas as pd
import requests
import schedule
import ta
from alpha_vantage.timeseries import TimeSeries
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from ta import add_all_ta_features

api_keys = ["IOBHACZOQONDVU8B", "STNP6RH4H5WMQKSN"]

def get_alpha_vantage_btc_history(api_keys):
    for api_key in api_keys:
        try:
            time.sleep(6)
            ts = TimeSeries(key=api_key)
            btc_data, meta_data = ts.get_daily(symbol='BTCUSD', outputsize='full')
            timestamps = list(btc_data.keys())
            open_prices = [float(btc_data[ts]['1. open']) for ts in timestamps]
            high_prices = [float(btc_data[ts]['2. high']) for ts in timestamps]
            low_prices = [float(btc_data[ts]['3. low']) for ts in timestamps]
            close_prices = [float(btc_data[ts]['4. close']) for ts in timestamps]
            volumes = [int(btc_data[ts]['5. volume']) for ts in timestamps]
            df = pd.DataFrame({
                "Timestamp": timestamps,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "volume": volumes,
                "close": close_prices
            })
            df["Date"] = pd.to_datetime(df["Timestamp"])
            df.to_csv("btc_price_data_alpha_vantage.csv", index=False)
            dfr = df[::-1]
            dfr.to_csv("btc_price_data_alpha_vantage_ful.csv", index=False)
            print("Saved full BTC price data from Alpha Vantage to btc_price_data_alpha_vantage_full.csv", flush=True)
            # Fetch current time from the internet
            response = requests.get("http://worldtimeapi.org/api/timezone/Etc/UTC")
            current_time = response.json()["datetime"]
            todays_date = pd.to_datetime(current_time[:10])  # Extract date from fetched time
            # Update the row for today's data (if it exists)
            if todays_date in dfr['Date'].values:
                # Get the current price from CoinDesk API
                current_price = get_current_btc_price() 
                print("assssissjdnuiks")
                dfr.loc[dfr['Date'] == todays_date, ['open', 'high', 'low', 'close']] = current_price
                dfr.loc[dfr['Date'] == todays_date, ['Date']] = todays_date  # Update the date
                return dfr
            else:
                # Fetch the current price from CoinDesk API
                current_price = get_current_btc_price()
                print("assssissjdnuiks")
                # Get yesterday's date
                yesterday_date = todays_date - pd.Timedelta(days=1)
                # Find the volume from yesterday
                yesterday_volume = dfr[dfr['Date'] == yesterday_date]['volume'].values[0]
                # Create a new row for today's data
                new_row = pd.DataFrame({
                    "Timestamp": pd.Timestamp.now().strftime('%Y-%m-%d'),
                    "open": dfr[dfr['Date'] == yesterday_date]['close'].values[0],
                    "high": current_price,
                    "low": current_price,
                    "volume": yesterday_volume,
                    "close": current_price,
                    "Date": todays_date
                }, index=[0])
                # Append the new row to the DataFrame
                dfr = pd.concat([dfr, new_row], ignore_index=True)
                dfr.to_csv("btc_price_data_alpha_vantage_ful.csv", index=False)
                return dfr
        except Exception as e:
            print(f"Error fetching BTC price using {api_key}: {str(e)}", flush=True)
            time.sleep(5)
    try:
        df = pd.read_csv("btc_price_data_alpha_vantage_ful.csv")
        df["Date"] = pd.to_datetime(df["Timestamp"].str.split().str[0])  # Only keep the date part
        # Fetch current time from the internet
        response = requests.get("http://worldtimeapi.org/api/timezone/Etc/UTC")
        current_time = response.json()["datetime"]
        todays_date = pd.to_datetime(current_time[:10])  # Extract date from fetched time
        # Check if today's date exists in the DataFrame
        if todays_date in df['Date'].values:
            # Update the row with today's date (replace existing row)
            df.loc[df['Date'] == todays_date, ['open', 'high', 'low', 'close']] = get_current_btc_price()
            df.loc[df['Date'] == todays_date, ['Date']] = todays_date  # Update the date
        else:
            # Append a new row with today's data
            current_price = get_current_btc_price()
            new_row = pd.DataFrame({
                "Timestamp": pd.Timestamp.now().strftime('%Y-%m-%d'),
                "open": current_price,
                "high": current_price,
                "low": current_price,
                "volume": 0,
                "close": current_price,
                "Date": todays_date
            }, index=[0])
            df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv("btc_price_data_alpha_vantage_ful.csv", index=False)
        print("Updated BTC price data in btc_price_data_alpha_vantage_ful.csv")
        return df
    except Exception as e:
        print(f"Error updating BTC data: {str(e)}")
        raise

def get_current_btc_price():
    try:
        url = "https://api.coindesk.com/v1/bpi/currentprice.json"
        time.sleep(1)
        response = requests.get(url)
        data = response.json()
        btc_price = data["bpi"]["USD"]["rate"]
        return float(btc_price.replace(",", ""))
    except Exception as e:
        return f"Error fetching BTC price: {str(e)}"

btc_history = get_alpha_vantage_btc_history(api_keys)

btc_data = pd.read_csv("btc_price_data_alpha_vantage_ful.csv")

# ... rest of your code ...

def predict_price_trend(btc_data, period=5):
    btc_data["SMA_20"] = btc_data["close"].rolling(window=20).mean()
    btc_data["EMA_50"] = btc_data["close"].ewm(span=50, adjust=False).mean()
    btc_data = add_all_ta_features(btc_data, "open", "high", "low", "close", "volume", fillna=True)
    btc_data["RSI"] = btc_data["momentum_rsi"]
    btc_data["EMA_12"] = btc_data["close"].ewm(span=12, adjust=False).mean()
    btc_data["EMA_26"] = btc_data["close"].ewm(span=26, adjust=False).mean()
    btc_data["MACD"] = btc_data["EMA_12"] - btc_data["EMA_26"]
    btc_data["Signal_Line"] = btc_data["MACD"].ewm(span=9, adjust=False).mean()
    btc_data["Upper_Band"], btc_data["Lower_Band"] = (
        btc_data["SMA_20"] + 2 * btc_data["close"].rolling(window=20).std(),
        btc_data["SMA_20"] - 2 * btc_data["close"].rolling(window=20).std(),
    )
    btc_data["ADX"] = ta.trend.ADXIndicator(
        btc_data["high"], btc_data["low"], btc_data["close"], window=14
    ).adx()
    btc_data["Stochastic_K"] = (
        (btc_data["close"] - btc_data["low"].rolling(window=14).min())
        / (btc_data["high"].rolling(window=14).max() - btc_data["low"].rolling(window=14).min())
    ) * 100
    X = btc_data[["SMA_20", "EMA_50", "RSI", "MACD", "ADX", "Stochastic_K"]]
    y = btc_data["close"]
    imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
    X = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=270, max_depth=14)
    model.fit(X_train, y_train)
    next_price = model.predict([[btc_data["SMA_20"].iloc[-1], btc_data["EMA_50"].iloc[-1], btc_data["RSI"].iloc[-1],
                                 btc_data["MACD"].iloc[-1], btc_data["ADX"].iloc[-1], btc_data["Stochastic_K"].iloc[-1]]])

    if period == 5:
        five_day_prices = [next_price[0]]
        for i in range(1, period):
            next_price = model.predict([[five_day_prices[i-1], btc_data["EMA_50"].iloc[-1], btc_data["RSI"].iloc[-1],
                                         btc_data["MACD"].iloc[-1], btc_data["ADX"].iloc[-1], btc_data["Stochastic_K"].iloc[-1]]])
            five_day_prices.append(next_price[0])
        return five_day_prices
    return next_price[0]


def getdateforprint():
    global time_infoo
    # Get the current time in UTC from the World Time API
    url = "http://worldtimeapi.org/api/timezone/Etc/UTC"
    response = requests.get(url)

    # If the request is successful, extract and print the date and time
    if response.status_code == 200:
        data = response.json()
        utc_datetime = data['datetime']

        # Split the date and time part
        date, time = utc_datetime.split("T")
        year, month, day = date.split("-")
        hour, minute, second = time.split(":")[:3]

        # Print the components
        time_infoo = f"Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Minute: {minute}, Second: {second}"
    else:
        print("Failed to fetch the UTC time")


def update_predictions():
    
    
    current_price = get_current_btc_price()
    btc_history = get_alpha_vantage_btc_history(api_keys)
    tomorrow_price = predict_price_trend(btc_data)
    five_day_prices = predict_price_trend(btc_data, period=5)
    tomorrow_price = float(tomorrow_price[0])
    getdateforprint()
    five_day_prices = [float(price) for price in five_day_prices]
    five_day_prices_with_index = enumerate(five_day_prices)
    price_comparison = ""
    recommendation = ""
    if tomorrow_price > current_price:
        percentage_increase = round(((tomorrow_price - current_price) / current_price) * 100, 2)
        price_comparison = f"Tomorrow's price is predicted to be {percentage_increase}% higher than today's price."
        if percentage_increase > 0.2:
            recommendation = "Buy 10% of your BTC amount."
        else:
            recommendation = "Buy a small percentage of your current BTC like 4 to 2 percent, or nothing."

    elif tomorrow_price < current_price:
        percentage_decrease = round(((current_price - tomorrow_price) / current_price) * 100, 2)
        price_comparison = f"Tomorrow's price is predicted to be {percentage_decrease}% lower than today's price."
        if percentage_decrease > 0.1:
            recommendation = "Sell 5% of your BTC."
        else:
            recommendation = "Do nothing or sell a really small percentage of BTC like 2% or do nothing."
    else:
        price_comparison = "Tomorrow's price is predicted to remain the same."
    global current_price_global, tomorrow_price_global, five_day_prices_with_index_global, price_comparison_global, recommendation_global
    current_price_global = current_price
    tomorrow_price_global = tomorrow_price
    five_day_prices_with_index_global = five_day_prices_with_index
    price_comparison_global = price_comparison
    recommendation_global = recommendation


schedule.every(60).minutes.do(update_predictions)



update_predictions()

class S(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET')
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            'current_price': current_price_global,
            'tomorrow_price': tomorrow_price_global,
            'price_comparison': price_comparison_global,
            'recommendation': recommendation_global,
            'timestamp_warning': f"**Note:** All timestamp at our historical data are in UTC time. So please check UTC time before making the prediction because that is also what the prediction time and day will be. updates every hour last updated time was: {time_infoo}"
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def do_POST(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET')
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "POST request received"}')

def run_server(server_class=HTTPServer, handler_class=S, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}')

    def handle_get_request():
        # Here is where the code for handling GET requests goes
        # You'll need to modify it to suit your specific needs
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET')
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            'current_price': current_price_global,
            'tomorrow_price': tomorrow_price_global,
            'price_comparison': price_comparison_global,
            'recommendation': recommendation_global
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))

    

    httpd.serve_forever()

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    from threading import Thread
    scheduler_thread = Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    run_server()
