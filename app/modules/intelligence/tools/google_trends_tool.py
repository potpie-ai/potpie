from typing import Type
from pytrends.request import TrendReq
from langchain.tools import BaseTool
from pydantic import BaseModel

class GoogleTrendsInput(BaseModel):
    query: str

class GoogleTrendsTool(BaseTool):
    name = "GoogleTrends"
    description = "Provides Google Trends data summary for a specific query over the past 7 days. Use this for understanding the current popularity of topics."
    args_schema: Type[BaseModel] = GoogleTrendsInput

    def _run(self, query: str) -> str:
        try:
            pytrends_instance = TrendReq(hl='en-US', tz=360)
            pytrends_instance.build_payload([query], cat=0, timeframe='now 7-d', geo='', gprop='')
            trends = pytrends_instance.interest_over_time()
            if trends.empty:
                return f"No trending data found for '{query}' in the past 7 days."
            else:
                trend_data = trends[query].tolist()
                recent_trend = trend_data[-1]
                return str(recent_trend)  # Ensure the trend is a string
        except Exception as e:
            return f"An error occurred while retrieving trends for '{query}': {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
