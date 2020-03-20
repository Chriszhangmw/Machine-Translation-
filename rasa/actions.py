from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import requests
from rasa_core.actions.action import Action
from rasa_core.events import SlotSet
from apixu.client import ApixuClient

class ActionWeather(Action):
	def name(self):
		return 'action_weather'
		
	def run(self, dispatcher, tracker, domain):
		import requests
		loc = tracker.get_slot('location')
		params = {
			'access_key': '7362a8270e7fe3f3505f0e2efaaaca51',
			'query': loc
		}
		api_result = requests.get('http://api.weatherstack.com/current', params)

		current = api_result.json()
		# print(api_response)
		#
		# api_key = 'f30bb30c8a6c354be6a7aad55a307777' #your apixu key
		# client = ApixuClient(api_key)
		#
		# loc = tracker.get_slot('location')
		# current = client.getcurrent(q=loc)

		# print(current)
		city = current['location']['name']

		temperature_c = current['current']['temperature']
		humidity = current['current']['humidity']
		wind_mph = current['current']['wind_speed']

		response = """the city  {} at the moment. The temperature is {} degrees, the humidity is {}% and the wind speed is {} mph.""".format(
			city, temperature_c, humidity, wind_mph)

		dispatcher.utter_message(response)
		return [SlotSet('location', loc)]

if __name__ == '__main__':
	import requests
	params = {
		'access_key': '7362a8270e7fe3f3505f0e2efaaaca51',
		'query': 'New York'
	}
	api_result = requests.get('http://api.weatherstack.com/current', params)

	api_response = api_result.json()
	print(api_response)
	api_key = '7362a8270e7fe3f3505f0e2efaaaca51'  # your apixu key
	client = ApixuClient(api_key)

	loc = 'London'
	current = client.current(q='London')
	print(current)
