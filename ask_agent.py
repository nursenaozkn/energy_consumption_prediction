from energy_agent_model import EnergyForecastAgent

agent = EnergyForecastAgent(model_dir="models")

date = input("Tahmin yapılacak tarihi girin (örn. 2027-03-12): ").strip()
result = agent.predict_daily(date)

print("\nTahmin Sonucu")
print("----------------")
print(f"Tarih: {result['date']}")
print(f"Tahmini toplam enerji tüketimi: {result['predicted_energy_consumption']:.2f}")
print(f"Tahmini yenilenebilir enerji oranı: %{result['predicted_renewable_share_percent']:.2f}")
print(f"Yenilenebilir enerji miktarı: {result['predicted_renewable_energy']:.2f}")
print(f"Yenilenebilir olmayan enerji miktarı: {result['predicted_nonrenewable_energy']:.2f}")
print(f"Enerji tüketimi %95 aralığı: {result['energy_95_interval'][0]:.2f} - {result['energy_95_interval'][1]:.2f}")
print(f"Yenilenebilir oran %95 aralığı: %{result['renewable_share_95_interval'][0]:.2f} - %{result['renewable_share_95_interval'][1]:.2f}")
print(f"Günlük enerji modeli güven seviyesi: {result['model_quality']['energy_reliability']}")
