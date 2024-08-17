from pytrends.request import TrendReq
from app.modules.intelligence.tools.tool_base import ToolBase

class GoogleTrendsTool(ToolBase):
    def __init__(self):
        self.name = "GoogleTrends"
        self.description = "Provides Google Trends data summary for a specific query over the past 7 days. Use this for understanding current popularity of topics."
        self.pytrends_instance = TrendReq(hl='en-US', tz=360)

    def run(self, query: str) -> str:
        try:
            self.pytrends_instance.build_payload([query], cat=0, timeframe='now 7-d', geo='', gprop='')
            trends = self.pytrends_instance.interest_over_time()
            if trends.empty:
                return f"No trending data found for '{query}' in the past 7 days."
            else:
                trend_data = trends[query].tolist()
                avg_trend = sum(trend_data) / len(trend_data)
                max_trend = max(trend_data)
                min_trend = min(trend_data)
                recent_trend = trend_data[-1]
                trend_direction = "increasing" if recent_trend > avg_trend else "decreasing"
                return (f"Trend summary for '{query}' over the past 7 days: "
                        f"Average: {avg_trend:.2f}, Max: {max_trend}, Min: {min_trend}, "
                        f"Recent: {recent_trend}, Overall trend: {trend_direction}")
        except Exception as e:
            return f"An error occurred while retrieving trends for '{query}': {str(e)}"