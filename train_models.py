from pprint import pprint
from energy_agent_model import train_and_save

# CSV dosyanızın adını/yolunu buraya yazın.
# Örnek: csv_path = "energy_data.csv"
csv_path = "energy_data_real.csv"

summary = train_and_save(csv_path=csv_path, model_dir="models")
pprint(summary)
