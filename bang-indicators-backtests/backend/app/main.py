from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
from auth import auth
from auth.models import Base, User
from dependencies import engine, get_pro_user, get_current_user
from fvg import (
    fetch_yfinance_data,
    identify_fair_value_gaps,
    backtest_fvg_strategy,
    calculate_strategy_metrics,
    plot_candlesticks_with_fvg_and_trades
)
import logging
import traceback  # Import traceback for detailed error information

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level to capture all log messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)  # Create a logger for this module

app = FastAPI(title="BANG Indicators and Backtests API")

# Create database tables
Base.metadata.create_all(bind=engine)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication router
app.include_router(auth.router)

class PlotRequest(BaseModel):
    indicator: str
    symbol: str
    interval: str
    start_date: str
    end_date: str

@app.post("/generate_plot")
def generate_plot(request: PlotRequest, user: User = Depends(get_current_user)):
    try:
        # Fetch data
        df = fetch_yfinance_data(request.symbol, request.interval, request.start_date, request.end_date)
        
        if df is None or df.empty:
            raise ValueError("No data fetched. Please check the symbol and date range.")

        # Identify Fair Value Gaps
        fvg_df = identify_fair_value_gaps(df)
        
        # Generate Plot with Open Gaps
        open_gaps = fvg_df.to_dict(orient='records')  # Assuming all gaps are open for plotting

        plot_image = plot_candlesticks_with_fvg_and_trades(df, open_gaps, "Fair Value Gaps")
        
        # Prepare Response
        response = {
            "plot_image": plot_image,
            "total_fvgs_identified": len(fvg_df)
        }
        
        return response

    except Exception as e:
        logger.error("An error occurred during plot generation:")
        logger.error(traceback.format_exc())  # Log the full stack trace
        raise HTTPException(status_code=500, detail="Internal Server Error")

class BacktestRequest(BaseModel):
    symbol: str
    interval: str
    start_date: str
    end_date: str

@app.post("/api/backtest")
def backtest(request: BacktestRequest, user: User = Depends(get_pro_user)):
    try:
        # Fetch data
        df = fetch_yfinance_data(request.symbol, request.interval, request.start_date, request.end_date)
        
        if df is None or df.empty:
            raise ValueError("No data fetched. Please check the symbol and date range.")
        
        # Identify Fair Value Gaps
        fvg_df = identify_fair_value_gaps(df, is_backtest=True)
        
        # Backtest Strategy
        trades_df, used_fvgs = backtest_fvg_strategy(df, fvg_df)
        
        # Calculate Metrics
        metrics = calculate_strategy_metrics(df, trades_df, request.start_date, request.end_date)
        
        # Prepare Response
        response = {
            "metrics": metrics,
            "closed_trades": trades_df.to_dict(orient='records'),
            "total_fvgs_identified": len(fvg_df),
            "fvgs_used_for_trades": len(used_fvgs),
            "total_trades_taken": len(trades_df)
        }
        
        return response

    except Exception as e:
        logger.error("An error occurred during backtesting:")
        logger.error(traceback.format_exc())  # Log the full stack trace
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/tickers")
def get_tickers():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'tickers.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError("Tickers CSV file not found.")

        df = pd.read_csv(csv_path)
        tickers = df.to_dict(orient='records')
        
        return {"tickers": tickers}
    
    except Exception as e:
        logger.error("An error occurred while fetching tickers:")
        logger.error(traceback.format_exc())  # Log the full stack trace
        raise HTTPException(status_code=500, detail="Internal Server Error")