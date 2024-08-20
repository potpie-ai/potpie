from typing import Type
from pytrends.request import TrendReq
from langchain.tools import BaseTool as LangchainToolBaseModel
from pydantic import BaseModel

class GoogleTrendsInput(BaseModel):
    query: str

class GoogleTrendsTool(LangchainToolBaseModel):
    name = "GoogleTrends"
    description = "Fetches Google Trends data for a query over the past 7 days."
    args_schema: Type[BaseModel] = GoogleTrendsInput

    def _run(self, query: str) -> str:
        try:
            pytrends_instance = TrendReq(hl='en-US', tz=360)
            pytrends_instance.build_payload([query], cat=0, timeframe='now 7-d', geo='', gprop='')
            trends = pytrends_instance.interest_over_time()

            if trends.empty:
                return f"No trends data found for '{query}' in the past 7 days."
            
            recent_trend = trends[query].tolist()[-1]
            return str(recent_trend).strip()
        except Exception as e:
            return f"Error retrieving Google Trends data: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
