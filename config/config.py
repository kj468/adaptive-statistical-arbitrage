"""Project constants for the Indian-equity pairs research workflow."""

from datetime import date, timedelta


STOCK_UNIVERSE: dict[str, dict[str, str]] = {
	"Banking & Financial Services": {
		"HDFCBANK.NS": "HDFC Bank",
		"ICICIBANK.NS": "ICICI Bank",
		"SBIN.NS": "State Bank of India",
		"AXISBANK.NS": "Axis Bank",
		"KOTAKBANK.NS": "Kotak Mahindra Bank",
		"INDUSINDBK.NS": "IndusInd Bank",
		"BANKBARODA.NS": "Bank of Baroda",
		"PNB.NS": "Punjab National Bank",
		"BAJFINANCE.NS": "Bajaj Finance",
	},
	"IT": {
		"TCS.NS": "TCS",
		"INFY.NS": "Infosys",
		"HCLTECH.NS": "HCL Technologies",
		"WIPRO.NS": "Wipro",
		"TECHM.NS": "Tech Mahindra",
		"LTIM.NS": "LTIMindtree",
		"MPHASIS.NS": "Mphasis",
		"PERSISTENT.NS": "Persistent Systems",
	},
	"Automobile": {
		"MARUTI.NS": "Maruti Suzuki",
		"M&M.NS": "Mahindra & Mahindra",
		"TATAMOTORS.NS": "Tata Motors",
		"BAJAJ-AUTO.NS": "Bajaj Auto",
		"EICHERMOT.NS": "Eicher Motors",
		"HEROMOTOCO.NS": "Hero MotoCorp",
		"TVSMOTOR.NS": "TVS Motor",
		"ASHOKLEY.NS": "Ashok Leyland",
	},
	"FMCG / Consumer": {
		"HINDUNILVR.NS": "Hindustan Unilever",
		"ITC.NS": "ITC",
		"NESTLEIND.NS": "Nestlé India",
		"BRITANNIA.NS": "Britannia Industries",
		"TATACONSUM.NS": "Tata Consumer Products",
		"DABUR.NS": "Dabur India",
		"GODREJCP.NS": "Godrej Consumer Products",
		"MARICO.NS": "Marico",
	},
	"Energy & Industrials": {
		"RELIANCE.NS": "Reliance Industries",
		"ONGC.NS": "ONGC",
		"COALINDIA.NS": "Coal India",
		"NTPC.NS": "NTPC",
		"POWERGRID.NS": "Power Grid",
		"BPCL.NS": "Bharat Petroleum",
		"IOC.NS": "Indian Oil",
		"GAIL.NS": "GAIL",
		"LT.NS": "Larsen & Toubro",
	},
	"Pharmaceuticals": {
		"SUNPHARMA.NS": "Sun Pharma",
		"DRREDDY.NS": "Dr. Reddy's Laboratories",
		"CIPLA.NS": "Cipla",
		"DIVISLAB.NS": "Divi's Laboratories",
		"LUPIN.NS": "Lupin",
		"AUROPHARMA.NS": "Aurobindo Pharma",
		"APOLLOHOSP.NS": "Apollo Hospitals",
		"TORNTPHARM.NS": "Torrent Pharma",
	},
}

TICKERS = tuple(
	ticker for sector_stocks in STOCK_UNIVERSE.values() for ticker in sector_stocks
)
TICKER_TO_SECTOR = {
	ticker: sector
	for sector, sector_stocks in STOCK_UNIVERSE.items()
	for ticker in sector_stocks
}

TRAIN_START_DATE = date(2016, 1, 1)
TRAIN_END_DATE = date(2023, 12, 31)
TEST_START_DATE = date(2024, 1, 1)
TEST_END_DATE = date(2025, 12, 31)
DEMO_START_DATE = date(2026, 1, 1)
DEMO_END_DATE = date(2026, 9, 30)

DATA_START_DATE = TRAIN_START_DATE
DATA_END_DATE = DEMO_END_DATE
DATA_END_DATE_EXCLUSIVE = DATA_END_DATE + timedelta(days=1)

ENTRY_ZSCORE = 2.0
EXIT_ZSCORE = 0.5
TRANSACTION_COST_PER_SIDE = 0.001
CAPITAL_PER_PAIR = 100_000
ROLLING_WINDOW = 60
SHARPE_RISK_FREE_RATE = 0.0
SORTINO_TARGET_RETURN = 0.0
TRADING_DAYS_PER_YEAR = 252
