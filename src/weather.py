import python_weather
import asyncio
from datetime import datetime

degree = u'\N{DEGREE SIGN}'

async def getweather():
    client = python_weather.Client(unit=python_weather.METRIC)
    try:
        weather = await client.get("Kingston, ON")
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        await client.close()
        return "Offline"

    today = datetime.today().strftime('%Y-%m-%d')

    for forecast in weather.forecasts:
        if str(forecast.date)[:10] == today:
            weather = f'Kingston, ON: {forecast.temperature}{degree}C'

    await client.close()
    return weather

async def weather():
    # Directly await the getweather() coroutine
    weather_info = await getweather()
    return weather_info


# Test harness
"""async def main():
    weather_info = await weather()
    print(weather_info)

# This is the entry point for running the test harness
if __name__ == "__main__":
    asyncio.run(main())"""