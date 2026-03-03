# tool_get_weather.py

# Import necessary libraries
from langchain.tools import tool
from pydantic import BaseModel, Field


class WeatherArgs(BaseModel):
    location: str = Field(..., description="The city name, e.g. San Francisco")
    units: str = Field("celsius", description="Units: 'celsius' or 'fahrenheit'")
    include_forecast: bool = Field(False, description="Include a 5-day forecast")


@tool(args_schema=WeatherArgs)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get the current weather for a given location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    temp = 22 if units.lower().startswith("c") else 72
    result = weather_data.get(location, f"Weather data not available for {location}")
    # If simulated entry exists, replace temperature line with computed temp
    if location in weather_data:
        result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
