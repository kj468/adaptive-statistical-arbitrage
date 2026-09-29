"""Generate unique unordered stock pairs within configured sectors."""

from itertools import combinations

from config.config import STOCK_UNIVERSE, TICKER_TO_SECTOR


def get_sector(ticker: str) -> str:
	"""Return a ticker's configured sector or raise a clear error."""
	try:
		return TICKER_TO_SECTOR[ticker]
	except KeyError as error:
		raise KeyError(f"Ticker {ticker!r} is not in the configured universe") from error


def generate_pairs(sector: str) -> list[tuple[str, str]]:
	"""Return each distinct pair in a sector exactly once."""
	try:
		tickers = tuple(STOCK_UNIVERSE[sector])
	except KeyError as error:
		raise KeyError(f"Unknown sector {sector!r}") from error
	return list(combinations(tickers, 2))