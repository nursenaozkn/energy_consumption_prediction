from pprint import pprint
from energy_agent_model import EnergyForecastAgent

agent = EnergyForecastAgent(model_dir="models")

# İstediğin tarihi burada değiştirebilirsin.
result = agent.predict_daily("2025-03-12")

pprint(result)
